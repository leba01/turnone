"""Training loop for BC policy.

Supports:
  - YAML config loading
  - Deterministic seeding (torch, numpy, random, CUDA)
  - Mixed precision (BF16 autocast on CUDA)
  - torch.compile() for speed
  - AdamW + CosineAnnealingLR
  - Cross-entropy with label smoothing
  - 3-term loss: action_A CE + action_B CE + lambda_tera * tera CE
  - Partial examples (action=-1 skipped)
  - Early stopping on val loss
  - Best-checkpoint saving + training curves + metadata

Reference: docs/PROJECT_BIBLE.md Sections 3.4-3.7
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
from turnone.models.bc_policy import BCPolicy
from turnone.data.dataset import build_dataloaders, Vocab
from turnone.eval.metrics import compute_bc_metrics


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
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    lambda_tera: float = 0.5,
    autoregressive: bool = False,
) -> float:
    """Train for one epoch. Returns average loss.

    Computes the 3-term loss:
      loss = CE(action_a) + CE(action_b) + lambda_tera * CE(tera)
    where action_a / action_b terms are skipped when the label is -1
    (mon fainted before acting).

    When autoregressive=True, uses forward_conditioned with teacher forcing:
      action_b is conditioned on ground-truth action_a.
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
        mask_a = batch["mask_a"].to(device, non_blocking=True)
        mask_b = batch["mask_b"].to(device, non_blocking=True)
        action_a = batch["action_a"].to(device, non_blocking=True)
        action_b = batch["action_b"].to(device, non_blocking=True)
        tera_label = batch["tera_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            if autoregressive:
                # Teacher forcing: condition on ground-truth action_a
                # For fainted mons, use padding index (SLOTS_PER_MON=16)
                action_a_cond = action_a.clone()
                action_a_cond[action_a == -1] = 16  # padding index
                out = model.forward_conditioned(
                    team_a, team_b, lead_a, lead_b, field_state,
                    mask_a, mask_b, action_a_cond,
                )
            else:
                out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

            # Build validity masks for partial examples
            valid_a = action_a != -1
            valid_b = action_b != -1

            loss = torch.tensor(0.0, device=device)

            if valid_a.any():
                loss = loss + criterion(out["logits_a"][valid_a], action_a[valid_a])
            if valid_b.any():
                loss = loss + criterion(out["logits_b"][valid_b], action_b[valid_b])

            loss = loss + lambda_tera * criterion(out["logits_tera"], tera_label)

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
    criterion: nn.Module,
    device: torch.device,
    lambda_tera: float = 0.5,
    autoregressive: bool = False,
) -> tuple[float, dict[str, np.ndarray]]:
    """Validate on a split.

    Returns
    -------
    avg_loss : float
        Mean loss (matching training loss computation for consistent early stopping).
    collected : dict[str, np.ndarray]
        Arrays for compute_bc_metrics:
          logits_a (N,16), logits_b (N,16), logits_tera (N,3),
          action_a (N,), action_b (N,), tera_label (N,),
          mask_a (N,16), mask_b (N,16).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits_a: list[np.ndarray] = []
    all_logits_b: list[np.ndarray] = []
    all_logits_tera: list[np.ndarray] = []
    all_action_a: list[np.ndarray] = []
    all_action_b: list[np.ndarray] = []
    all_tera_label: list[np.ndarray] = []
    all_mask_a: list[np.ndarray] = []
    all_mask_b: list[np.ndarray] = []

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        lead_a = batch["lead_a"].to(device, non_blocking=True)
        lead_b = batch["lead_b"].to(device, non_blocking=True)
        field_state = batch["field_state"].to(device, non_blocking=True)
        mask_a = batch["mask_a"].to(device, non_blocking=True)
        mask_b = batch["mask_b"].to(device, non_blocking=True)
        action_a = batch["action_a"].to(device, non_blocking=True)
        action_b = batch["action_b"].to(device, non_blocking=True)
        tera_label = batch["tera_label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            if autoregressive:
                action_a_cond = action_a.clone()
                action_a_cond[action_a == -1] = 16
                out = model.forward_conditioned(
                    team_a, team_b, lead_a, lead_b, field_state,
                    mask_a, mask_b, action_a_cond,
                )
            else:
                out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

            valid_a = action_a != -1
            valid_b = action_b != -1

            loss = torch.tensor(0.0, device=device)
            if valid_a.any():
                loss = loss + criterion(out["logits_a"][valid_a], action_a[valid_a])
            if valid_b.any():
                loss = loss + criterion(out["logits_b"][valid_b], action_b[valid_b])
            loss = loss + lambda_tera * criterion(out["logits_tera"], tera_label)

        total_loss += loss.item()
        n_batches += 1

        # Collect raw logits in FP32 for metrics
        all_logits_a.append(out["logits_a"].float().cpu().numpy())
        all_logits_b.append(out["logits_b"].float().cpu().numpy())
        all_logits_tera.append(out["logits_tera"].float().cpu().numpy())
        all_action_a.append(batch["action_a"].numpy())
        all_action_b.append(batch["action_b"].numpy())
        all_tera_label.append(batch["tera_label"].numpy())
        all_mask_a.append(batch["mask_a"].numpy())
        all_mask_b.append(batch["mask_b"].numpy())

    avg_loss = total_loss / max(n_batches, 1)
    collected = {
        "logits_a": np.concatenate(all_logits_a, axis=0),
        "logits_b": np.concatenate(all_logits_b, axis=0),
        "logits_tera": np.concatenate(all_logits_tera, axis=0),
        "action_a": np.concatenate(all_action_a, axis=0),
        "action_b": np.concatenate(all_action_b, axis=0),
        "tera_label": np.concatenate(all_tera_label, axis=0),
        "mask_a": np.concatenate(all_mask_a, axis=0),
        "mask_b": np.concatenate(all_mask_b, axis=0),
    }
    return avg_loss, collected


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
        require_both_actions=cfg_train.get("require_both_actions", False),
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
    autoregressive = cfg_train.get("autoregressive", False)
    model = BCPolicy(vocab.vocab_sizes, encoder_cfg, autoregressive=autoregressive)
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
        _ma = torch.ones(_B, 16, dtype=torch.bool, device=device)
        _mb = torch.ones(_B, 16, dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            if autoregressive:
                _aa = torch.zeros(_B, dtype=torch.long, device=device)
                compiled_model.forward_conditioned(
                    _ta, _tb, _la, _lb, _fs, _ma, _mb, _aa)
                del _aa
            else:
                compiled_model(_ta, _tb, _la, _lb, _fs, _ma, _mb)
        del _ta, _tb, _la, _lb, _fs, _ma, _mb
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
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg_train["label_smoothing"],
    )

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # --- Training loop ---
    lambda_tera = cfg_train.get("lambda_tera", 0.5)
    patience = cfg_train["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = -1

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "top1_avg": [],
    }

    best_ckpt_path = out_dir / "best.pt"

    header = (
        f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  "
        f"{'Top1 Avg':>8}  {'LR':>10}  {'Best':>5}  {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            compiled_model, train_loader, optimizer, criterion,
            scaler, device, lambda_tera=lambda_tera,
            autoregressive=autoregressive,
        )

        # Validate
        val_loss, val_outputs = validate(
            compiled_model, val_loader, criterion, device,
            lambda_tera=lambda_tera,
            autoregressive=autoregressive,
        )

        # Compute BC metrics on validation outputs
        bc_metrics = compute_bc_metrics(
            logits_a=val_outputs["logits_a"],
            logits_b=val_outputs["logits_b"],
            logits_tera=val_outputs["logits_tera"],
            action_a=val_outputs["action_a"],
            action_b=val_outputs["action_b"],
            tera_label=val_outputs["tera_label"],
            mask_a=val_outputs["mask_a"],
            mask_b=val_outputs["mask_b"],
        )
        top1_avg = bc_metrics["top1_avg"]

        # LR step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["top1_avg"].append(top1_avg)

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
                    "autoregressive": autoregressive,
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
            f"{top1_avg:8.1%}  {current_lr:10.2e}  {mark:>5}  {dt:5.1f}s"
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

    parser = argparse.ArgumentParser(description="Train BC policy")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.out_dir)
