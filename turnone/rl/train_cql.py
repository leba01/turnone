"""Training loop for CQL (Conservative Q-Learning) policy.

Horizon-1 CQL: Q(s, a) = E[r | s, a_me], no bootstrapping.
Factored Q: Q(s, slot_a, slot_b, tera) = Q_a(s,slot_a) + Q_b(s,slot_b) + Q_tera(s,tera).

Loss = MSE(Q_data, reward) + alpha * CQL_penalty
CQL_penalty = logsumexp over valid joint actions - Q(s, a_data)

Reference: Kumar et al., "Conservative Q-Learning for Offline RL" (NeurIPS 2020)
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
from turnone.models.cql_qvalue import QLearner
from turnone.data.dataset import build_dataloaders, Vocab
from turnone.data.action_space import NUM_TERA


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> dict[str, Any]:
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CQL loss helpers
# ---------------------------------------------------------------------------

def _compute_cql_loss(
    q_a: torch.Tensor,        # (B, 16)
    q_b: torch.Tensor,        # (B, 16)
    q_tera: torch.Tensor,     # (B, 3)
    action_a: torch.Tensor,   # (B,) int
    action_b: torch.Tensor,   # (B,) int
    tera_label: torch.Tensor, # (B,) int
    reward: torch.Tensor,     # (B,)
    strat_mask_a: torch.Tensor,  # (B, 16) bool
    strat_mask_b: torch.Tensor,  # (B, 16) bool
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CQL loss: TD loss + alpha * CQL penalty.

    TD loss: MSE between Q(s, a_data) and reward.
    CQL penalty (joint): logsumexp over valid (sa, sb, t) triples - Q(s, a_data).
    """
    B = q_a.size(0)
    batch_idx = torch.arange(B, device=q_a.device)

    # Q(s, a_data) = Q_a[action_a] + Q_b[action_b] + Q_tera[tera_label]
    q_data = (q_a[batch_idx, action_a]
              + q_b[batch_idx, action_b]
              + q_tera[batch_idx, tera_label])

    # TD loss (MSE)
    td_loss = ((q_data - reward) ** 2).mean()

    # CQL penalty: joint logsumexp over valid actions
    # For each example, enumerate valid (sa, sb, t) and compute Q_joint
    # Use masking: q_a for invalid slots = -inf, same for q_b
    _NEG_INF = -1e9
    q_a_masked = q_a.clone()
    q_a_masked[~strat_mask_a] = _NEG_INF
    q_b_masked = q_b.clone()
    q_b_masked[~strat_mask_b] = _NEG_INF

    # Joint logsumexp via factored trick:
    # sum_joint exp(Q_a[sa] + Q_b[sb] + Q_tera[t])
    #   = (sum_sa exp(Q_a[sa])) * (sum_sb exp(Q_b[sb])) * (sum_t exp(Q_tera[t]))
    # So logsumexp_joint = logsumexp_a + logsumexp_b + logsumexp_tera
    #
    # NOTE: This is a valid upper bound and is the standard factored CQL approach.
    # It equals the exact joint logsumexp when Q is additive (which ours is).
    lse_a = torch.logsumexp(q_a_masked, dim=1)     # (B,)
    lse_b = torch.logsumexp(q_b_masked, dim=1)     # (B,)
    lse_tera = torch.logsumexp(q_tera, dim=1)       # (B,)
    lse_joint = lse_a + lse_b + lse_tera             # (B,)

    cql_penalty = (lse_joint - q_data).mean()

    total_loss = td_loss + alpha * cql_penalty

    metrics = {
        "td_loss": td_loss.item(),
        "cql_penalty": cql_penalty.item(),
        "q_data_mean": q_data.mean().item(),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    alpha: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch. Returns (avg_loss, avg_metrics)."""
    model.train()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        lead_a = batch["lead_a"].to(device, non_blocking=True)
        lead_b = batch["lead_b"].to(device, non_blocking=True)
        field_state = batch["field_state"].to(device, non_blocking=True)
        action_a = batch["action_a"].to(device, non_blocking=True)
        action_b = batch["action_b"].to(device, non_blocking=True)
        tera_label = batch["tera_label"].to(device, non_blocking=True)
        reward = batch["reward"].to(device, non_blocking=True)
        strat_mask_a = batch["strategic_mask_a"].to(device, non_blocking=True)
        strat_mask_b = batch["strategic_mask_b"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            out = model(team_a, team_b, lead_a, lead_b, field_state)
            loss, metrics = _compute_cql_loss(
                out["q_a"], out["q_b"], out["q_tera"],
                action_a, action_b, tera_label,
                reward, strat_mask_a, strat_mask_b,
                alpha=alpha,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return total_loss / n, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Validate on a split. Returns (avg_loss, avg_metrics)."""
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)
        lead_a = batch["lead_a"].to(device, non_blocking=True)
        lead_b = batch["lead_b"].to(device, non_blocking=True)
        field_state = batch["field_state"].to(device, non_blocking=True)
        action_a = batch["action_a"].to(device, non_blocking=True)
        action_b = batch["action_b"].to(device, non_blocking=True)
        tera_label = batch["tera_label"].to(device, non_blocking=True)
        reward = batch["reward"].to(device, non_blocking=True)
        strat_mask_a = batch["strategic_mask_a"].to(device, non_blocking=True)
        strat_mask_b = batch["strategic_mask_b"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            out = model(team_a, team_b, lead_a, lead_b, field_state)
            loss, metrics = _compute_cql_loss(
                out["q_a"], out["q_b"], out["q_tera"],
                action_a, action_b, tera_label,
                reward, strat_mask_a, strat_mask_b,
                alpha=alpha,
            )

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return total_loss / n, avg_metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any], out_dir: str | Path) -> Path:
    """Full CQL training loop.

    Returns path to best checkpoint.
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
        canonicalize_targets=cfg_train.get("canonicalize_targets", True),
    )
    print(f"Vocab: {vocab}")
    print(f"Train: {len(train_loader.dataset):,} examples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset):,} examples")

    vocab.save(out_dir / "vocab.json")

    # --- Model ---
    encoder_cfg = EncoderConfig(
        d_model=cfg_model["d_model"],
        n_layers=cfg_model["n_layers"],
        n_heads=cfg_model["n_heads"],
        d_ff=cfg_model["d_ff"],
        dropout=cfg_model["dropout"],
    )
    model = QLearner(vocab.vocab_sizes, encoder_cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # torch.compile
    compiled_model = model
    try:
        compiled_model = torch.compile(model)
        _B = 2
        _ta = torch.zeros(_B, 6, 8, dtype=torch.long, device=device)
        _tb = torch.zeros(_B, 6, 8, dtype=torch.long, device=device)
        _la = torch.tensor([[0, 1]] * _B, dtype=torch.long, device=device)
        _lb = torch.tensor([[0, 1]] * _B, dtype=torch.long, device=device)
        _fs = torch.zeros(_B, 5, device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            compiled_model(_ta, _tb, _la, _lb, _fs)
        del _ta, _tb, _la, _lb, _fs
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

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # --- Training loop ---
    alpha = cfg_train.get("alpha", 1.0)
    patience = cfg_train["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = -1

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "td_loss": [], "cql_penalty": [], "q_data_mean": [],
    }

    best_ckpt_path = out_dir / "best.pt"

    header = (
        f"{'Epoch':>5}  {'Train':>10}  {'Val':>10}  "
        f"{'TD':>8}  {'CQL':>8}  {'Q_data':>8}  {'LR':>10}  {'Best':>5}  {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            compiled_model, train_loader, optimizer, scaler, device,
            alpha=alpha,
        )

        val_loss, val_metrics = validate(
            compiled_model, val_loader, device, alpha=alpha,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for k in ["td_loss", "cql_penalty", "q_data_mean"]:
            history[k].append(val_metrics.get(k, 0.0))

        # Check improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

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
        td = val_metrics.get("td_loss", 0.0)
        cql = val_metrics.get("cql_penalty", 0.0)
        qd = val_metrics.get("q_data_mean", 0.0)
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  "
            f"{td:8.4f}  {cql:8.4f}  {qd:8.4f}  {current_lr:10.2e}  {mark:>5}  {dt:5.1f}s"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {best_ckpt_path}")

    # Save curves + metadata
    with open(out_dir / "training_curves.json", "w") as f:
        json.dump(history, f, indent=2)

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
        "torch_version": torch.__version__,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CQL Q-network")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.out_dir)
