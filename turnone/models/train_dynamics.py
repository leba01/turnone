"""Training loop for dynamics model.

Predicts turn-1 outcomes (HP changes, KOs, field state changes) from
state + joint actions.

Supports:
  - YAML config loading
  - Deterministic seeding (torch, numpy, random, CUDA)
  - Mixed precision (BF16 autocast on CUDA)
  - torch.compile() for speed
  - AdamW + CosineAnnealingLR
  - Decomposed loss: MSE(hp) + lambda_ko * BCE(ko)
    + lambda_field * [CE(weather) + CE(terrain) + BCE(binary_field)]
  - Proper action remapping (-1 -> 16 for fainted mons)
  - Early stopping on val loss
  - Best-checkpoint saving + training curves + metadata

Reference: docs/PROJECT_BIBLE.md Sections 4.1-4.3
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from turnone.models.encoder import EncoderConfig
from turnone.models.dynamics import DynamicsModel, remap_actions
from turnone.data.dataset import build_dataloaders, Vocab


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dict."""
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed torch, numpy, random, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Git hash
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return short git hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Action remapping helper
# ---------------------------------------------------------------------------

def _remap_batch_actions(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move action tensors to device and remap -1 -> 16 (no-action).

    Fainted mons have action=-1 in the dataset. The dynamics model has a
    dedicated no-action embedding at index 16.
    """
    action_keys = ("action_a", "action_b", "opp_action_a", "opp_action_b")
    result = {}
    for key in action_keys:
        val = batch[key].to(device, non_blocking=True)
        result[key] = remap_actions(val)
    return result


# ---------------------------------------------------------------------------
# Field target decomposition
# ---------------------------------------------------------------------------

def _decompose_field_targets(field_after: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose field_after (B, 5) into categorical and binary targets.

    field_state = [weather_idx, terrain_idx, trick_room, tailwind_p1, tailwind_p2]

    Returns:
        weather_target: (B,) LongTensor, weather class index (0-4)
        terrain_target: (B,) LongTensor, terrain class index (0-4)
        binary_target: (B, 3) FloatTensor, [trick_room, tw_p1, tw_p2]
    """
    weather_target = field_after[:, 0].long()
    terrain_target = field_after[:, 1].long()
    binary_target = field_after[:, 2:5]  # trick_room, tw_p1, tw_p2
    return weather_target, terrain_target, binary_target


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    hp_criterion: nn.Module,
    ko_criterion: nn.Module,
    weather_criterion: nn.Module,
    terrain_criterion: nn.Module,
    binary_field_criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    lambda_ko: float = 3.0,
    lambda_field: float = 0.5,
) -> float:
    """Train for one epoch. Returns average loss.

    Computes the decomposed loss:
      loss = MSE(hp)
           + lambda_ko * BCE(ko)
           + lambda_field * [CE(weather) + CE(terrain) + BCE(binary_field)]
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        lead_a = batch["lead_a"].to(device, non_blocking=True)
        lead_b = batch["lead_b"].to(device, non_blocking=True)
        field_state = batch["field_state"].to(device, non_blocking=True)
        tera_flag = batch["tera_label"].to(device, non_blocking=True)
        opp_tera_flag = batch["opp_tera_label"].to(device, non_blocking=True)

        # Targets
        hp_delta = batch["hp_delta"].to(device, non_blocking=True)
        ko_flags = batch["ko_flags"].to(device, non_blocking=True)
        field_after = batch["field_after"].to(device, non_blocking=True)

        # Decompose field targets
        weather_target, terrain_target, binary_target = _decompose_field_targets(field_after)

        # Remap actions (fainted mons: -1 -> 16)
        remapped = _remap_batch_actions(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            hp_pred, ko_logits, field_logits = model(
                team_a, team_b, lead_a, lead_b, field_state,
                remapped["action_a"], remapped["action_b"], tera_flag,
                remapped["opp_action_a"], remapped["opp_action_b"], opp_tera_flag,
            )

            hp_loss = hp_criterion(hp_pred, hp_delta)
            ko_loss = ko_criterion(ko_logits, ko_flags)
            weather_loss = weather_criterion(field_logits.weather, weather_target)
            terrain_loss = terrain_criterion(field_logits.terrain, terrain_target)
            binary_field_loss = binary_field_criterion(field_logits.binary, binary_target)

            field_loss = weather_loss + terrain_loss + binary_field_loss
            loss = hp_loss + lambda_ko * ko_loss + lambda_field * field_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    hp_criterion: nn.Module,
    ko_criterion: nn.Module,
    weather_criterion: nn.Module,
    terrain_criterion: nn.Module,
    binary_field_criterion: nn.Module,
    device: torch.device,
    lambda_ko: float = 3.0,
    lambda_field: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Validate on a split.

    Returns
    -------
    avg_loss : float
        Mean loss (matching training loss computation for consistent early stopping).
    metrics : dict[str, float]
        hp_mae, ko_acc, weather_acc, terrain_acc, binary_field_acc for the epoch table.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Accumulators for metrics
    all_hp_ae: list[np.ndarray] = []        # absolute errors
    all_ko_correct: list[np.ndarray] = []   # per-slot correctness
    all_weather_correct: list[np.ndarray] = []
    all_terrain_correct: list[np.ndarray] = []
    all_binary_correct: list[np.ndarray] = []

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        lead_a = batch["lead_a"].to(device, non_blocking=True)
        lead_b = batch["lead_b"].to(device, non_blocking=True)
        field_state = batch["field_state"].to(device, non_blocking=True)
        tera_flag = batch["tera_label"].to(device, non_blocking=True)
        opp_tera_flag = batch["opp_tera_label"].to(device, non_blocking=True)

        hp_delta = batch["hp_delta"].to(device, non_blocking=True)
        ko_flags = batch["ko_flags"].to(device, non_blocking=True)
        field_after = batch["field_after"].to(device, non_blocking=True)

        weather_target, terrain_target, binary_target = _decompose_field_targets(field_after)
        remapped = _remap_batch_actions(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            hp_pred, ko_logits, field_logits = model(
                team_a, team_b, lead_a, lead_b, field_state,
                remapped["action_a"], remapped["action_b"], tera_flag,
                remapped["opp_action_a"], remapped["opp_action_b"], opp_tera_flag,
            )

            hp_loss = hp_criterion(hp_pred, hp_delta)
            ko_loss = ko_criterion(ko_logits, ko_flags)
            weather_loss = weather_criterion(field_logits.weather, weather_target)
            terrain_loss = terrain_criterion(field_logits.terrain, terrain_target)
            binary_field_loss = binary_field_criterion(field_logits.binary, binary_target)

            field_loss = weather_loss + terrain_loss + binary_field_loss
            loss = hp_loss + lambda_ko * ko_loss + lambda_field * field_loss

        total_loss += loss.item()
        n_batches += 1

        # Collect metrics in FP32
        hp_pred_f = hp_pred.float().cpu()
        ko_logits_f = ko_logits.float().cpu()
        hp_delta_f = hp_delta.float().cpu()
        ko_flags_f = ko_flags.float().cpu()

        weather_logits_f = field_logits.weather.float().cpu()
        terrain_logits_f = field_logits.terrain.float().cpu()
        binary_logits_f = field_logits.binary.float().cpu()

        # HP MAE: per-slot absolute error
        all_hp_ae.append((hp_pred_f - hp_delta_f).abs().numpy())

        # KO Acc: sigmoid(logits) > 0.5 vs binary flags
        ko_preds = (torch.sigmoid(ko_logits_f) > 0.5).float()
        all_ko_correct.append((ko_preds == ko_flags_f).numpy())

        # Weather accuracy: argmax vs target
        weather_preds = weather_logits_f.argmax(dim=-1)
        all_weather_correct.append((weather_preds == weather_target.cpu()).numpy())

        # Terrain accuracy: argmax vs target
        terrain_preds = terrain_logits_f.argmax(dim=-1)
        all_terrain_correct.append((terrain_preds == terrain_target.cpu()).numpy())

        # Binary field accuracy: sigmoid > 0.5 vs target
        binary_preds = (torch.sigmoid(binary_logits_f) > 0.5).float()
        all_binary_correct.append((binary_preds == binary_target.cpu()).numpy())

    avg_loss = total_loss / max(n_batches, 1)

    hp_mae = float(np.concatenate(all_hp_ae, axis=0).mean())
    ko_acc = float(np.concatenate(all_ko_correct, axis=0).mean())
    weather_acc = float(np.concatenate(all_weather_correct, axis=0).mean())
    terrain_acc = float(np.concatenate(all_terrain_correct, axis=0).mean())
    binary_field_acc = float(np.concatenate(all_binary_correct, axis=0).mean())

    metrics = {
        "hp_mae": hp_mae,
        "ko_acc": ko_acc,
        "weather_acc": weather_acc,
        "terrain_acc": terrain_acc,
        "binary_field_acc": binary_field_acc,
    }
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any], out_dir: str | Path) -> Path:
    """Full training loop.

    Parameters
    ----------
    config : dict
        Loaded YAML config with 'model', 'training', 'data' sections.
    out_dir : path
        Output directory for checkpoints, curves, and metadata.

    Returns
    -------
    Path to best checkpoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_data = config["data"]

    seed = cfg_train["seed"]
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    split_dir = cfg_data["split_dir"]
    print(f"Loading data from {split_dir} ...")
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        split_dir=split_dir,
        batch_size=cfg_train["batch_size"],
        num_workers=cfg_train["num_workers"],
        vocab_path=cfg_data.get("vocab_path"),
        require_both_actions=cfg_train.get("require_both_actions", True),
        canonicalize_targets=cfg_train.get("canonicalize_targets", False),
    )
    print(f"Vocab: {vocab}")
    print(f"Train: {len(train_loader.dataset):,} examples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset):,} examples, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset):,} examples, {len(test_loader)} batches")

    # Save vocab alongside checkpoint
    vocab.save(out_dir / "vocab.json")

    # --- Model ---
    encoder_cfg = EncoderConfig(
        d_model=cfg_model["d_model"],
        n_layers=cfg_model["n_layers"],
        n_heads=cfg_model["n_heads"],
        d_ff=cfg_model["d_ff"],
        dropout=cfg_model["dropout"],
    )
    d_action = cfg_model.get("d_action", 32)
    d_hidden = cfg_model.get("d_hidden", 256)
    n_mlp_layers = cfg_model.get("n_mlp_layers", 3)
    model_dropout = cfg_model.get("dropout", 0.1)
    action_cross_attn = cfg_model.get("action_cross_attn", False)
    action_attn_heads = cfg_model.get("action_attn_heads", 2)
    action_attn_layers = cfg_model.get("action_attn_layers", 1)

    model = DynamicsModel(
        vocab.vocab_sizes, encoder_cfg,
        d_action=d_action,
        d_hidden=d_hidden,
        n_mlp_layers=n_mlp_layers,
        dropout=model_dropout,
        action_cross_attn=action_cross_attn,
        action_attn_heads=action_attn_heads,
        action_attn_layers=action_attn_layers,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # torch.compile for speed (graceful fallback if unavailable)
    compiled_model = model  # keep reference to original for state_dict saving
    try:
        compiled_model = torch.compile(model)
        # Force a test compilation with a dummy forward pass
        _B = 2
        _ta = torch.zeros(_B, 6, 8, dtype=torch.long, device=device)
        _tb = torch.zeros(_B, 6, 8, dtype=torch.long, device=device)
        _la = torch.tensor([[0, 1]] * _B, dtype=torch.long, device=device)
        _lb = torch.tensor([[0, 1]] * _B, dtype=torch.long, device=device)
        _fs = torch.zeros(_B, 5, device=device)
        _aa = torch.zeros(_B, dtype=torch.long, device=device)
        _ab = torch.zeros(_B, dtype=torch.long, device=device)
        _tf = torch.zeros(_B, dtype=torch.long, device=device)
        _oa = torch.zeros(_B, dtype=torch.long, device=device)
        _ob = torch.zeros(_B, dtype=torch.long, device=device)
        _ot = torch.zeros(_B, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            compiled_model(_ta, _tb, _la, _lb, _fs, _aa, _ab, _tf, _oa, _ob, _ot)
        del _ta, _tb, _la, _lb, _fs, _aa, _ab, _tf, _oa, _ob, _ot
        print("torch.compile() applied")
    except Exception as e:
        compiled_model = model
        print(f"torch.compile() skipped ({type(e).__name__}: {e})")

    # --- Optimizer + scheduler ---
    optimizer = AdamW(
        compiled_model.parameters(),
        lr=cfg_train["lr"],
        weight_decay=cfg_train["weight_decay"],
    )
    max_epochs = cfg_train["max_epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # --- Loss ---
    hp_criterion = nn.MSELoss()
    ko_criterion = nn.BCEWithLogitsLoss()
    weather_criterion = nn.CrossEntropyLoss()
    terrain_criterion = nn.CrossEntropyLoss()
    binary_field_criterion = nn.BCEWithLogitsLoss()

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # --- Training loop ---
    lambda_ko = cfg_train.get("lambda_ko", 3.0)
    lambda_field = cfg_train.get("lambda_field", 0.5)
    patience = cfg_train["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = -1

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "hp_mae": [],
        "ko_acc": [],
        "weather_acc": [],
        "terrain_acc": [],
        "binary_field_acc": [],
    }

    best_ckpt_path = out_dir / "best.pt"

    header = (
        f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  "
        f"{'HP MAE':>8}  {'KO Acc':>8}  {'Wea Acc':>8}  {'Ter Acc':>8}  "
        f"{'LR':>10}  {'Best':>5}  {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            compiled_model, train_loader, optimizer,
            hp_criterion, ko_criterion,
            weather_criterion, terrain_criterion, binary_field_criterion,
            scaler, device,
            lambda_ko=lambda_ko, lambda_field=lambda_field,
        )

        # Validate
        val_loss, val_metrics = validate(
            compiled_model, val_loader,
            hp_criterion, ko_criterion,
            weather_criterion, terrain_criterion, binary_field_criterion,
            device,
            lambda_ko=lambda_ko, lambda_field=lambda_field,
        )

        hp_mae = val_metrics["hp_mae"]
        ko_acc = val_metrics["ko_acc"]
        weather_acc = val_metrics["weather_acc"]
        terrain_acc = val_metrics["terrain_acc"]

        # LR step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["hp_mae"].append(hp_mae)
        history["ko_acc"].append(ko_acc)
        history["weather_acc"].append(weather_acc)
        history["terrain_acc"].append(terrain_acc)
        history["binary_field_acc"].append(val_metrics["binary_field_acc"])

        # Check improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint (unwrap compiled model -> use original model)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_sizes": vocab.vocab_sizes,
                    "encoder_config": {
                        "d_model": encoder_cfg.d_model,
                        "n_layers": encoder_cfg.n_layers,
                        "n_heads": encoder_cfg.n_heads,
                        "d_ff": encoder_cfg.d_ff,
                        "dropout": encoder_cfg.dropout,
                    },
                    "d_action": d_action,
                    "d_hidden": d_hidden,
                    "n_mlp_layers": n_mlp_layers,
                    "dropout": model_dropout,
                    "action_cross_attn": action_cross_attn,
                    "action_attn_heads": action_attn_heads,
                    "action_attn_layers": action_attn_layers,
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        dt = time.time() - t0
        mark = "*" if is_best else ""
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  "
            f"{hp_mae:8.4f}  {ko_acc:8.1%}  {weather_acc:8.1%}  {terrain_acc:8.1%}  "
            f"{current_lr:10.2e}  {mark:>5}  {dt:5.1f}s"
        )

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {best_ckpt_path}")

    # --- Save training curves ---
    curves_path = out_dir / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training curves saved to {curves_path}")

    # --- Save run metadata ---
    metadata = {
        "config": config,
        "seed": seed,
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(history["train_loss"]),
        "vocab_sizes": vocab.vocab_sizes,
        "train_examples": len(train_loader.dataset),
        "val_examples": len(val_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved to {meta_path}")

    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train dynamics model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.out_dir)
