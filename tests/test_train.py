"""Tests for BC training loop.

All tests use tiny synthetic datasets and run on CPU.
No GPU or real data required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from turnone.models.train import load_config, seed_everything, train_one_epoch, validate
from turnone.models.encoder import EncoderConfig
from turnone.models.bc_policy import BCPolicy


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------

TINY_VOCAB = {"species": 20, "item": 10, "ability": 15, "tera_type": 5, "move": 30}
TINY_CFG = EncoderConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0)


class TinyDataset(Dataset):
    """Minimal dataset matching Turn1Dataset output format."""

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    @staticmethod
    def _random_team() -> torch.Tensor:
        """Generate a (6, 8) team tensor with per-field valid ranges."""
        # Fields: species(20), item(10), ability(15), tera_type(5), move x4(30)
        cols = [
            torch.randint(0, TINY_VOCAB["species"], (6, 1)),
            torch.randint(0, TINY_VOCAB["item"], (6, 1)),
            torch.randint(0, TINY_VOCAB["ability"], (6, 1)),
            torch.randint(0, TINY_VOCAB["tera_type"], (6, 1)),
            torch.randint(0, TINY_VOCAB["move"], (6, 1)),
            torch.randint(0, TINY_VOCAB["move"], (6, 1)),
            torch.randint(0, TINY_VOCAB["move"], (6, 1)),
            torch.randint(0, TINY_VOCAB["move"], (6, 1)),
        ]
        return torch.cat(cols, dim=1)

    def __getitem__(self, idx: int) -> dict:
        return {
            "team_a": self._random_team(),
            "team_b": self._random_team(),
            "lead_a": torch.tensor([0, 1]),
            "lead_b": torch.tensor([0, 1]),
            "field_state": torch.randn(5),
            "action_a": 0 if idx % 5 != 0 else -1,  # some -1s
            "action_b": 1 if idx % 7 != 0 else -1,
            "mask_a": torch.ones(16, dtype=torch.bool),
            "mask_b": torch.ones(16, dtype=torch.bool),
            "tera_label": idx % 3,
            "opp_action_a": 0,
            "opp_action_b": 1,
            "opp_mask_a": torch.ones(16, dtype=torch.bool),
            "opp_mask_b": torch.ones(16, dtype=torch.bool),
            "opp_tera_label": 0,
            "hp_delta": torch.randn(4),
            "ko_flags": torch.zeros(4),
            "field_after": torch.randn(5),
        }


def _make_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    """Create a tiny DataLoader."""
    return DataLoader(TinyDataset(n), batch_size=batch_size, shuffle=False)


def _make_model() -> BCPolicy:
    """Create a tiny BCPolicy."""
    return BCPolicy(TINY_VOCAB, TINY_CFG)


# ===========================================================================
# TestSeedEverything
# ===========================================================================


class TestSeedEverything:
    """Tests for seed_everything."""

    def test_deterministic(self):
        """Same seed produces the same random tensor."""
        seed_everything(42)
        t1 = torch.randn(5)

        seed_everything(42)
        t2 = torch.randn(5)

        assert torch.allclose(t1, t2), "Tensors should be identical with same seed"


# ===========================================================================
# TestTrainOneEpoch
# ===========================================================================


class TestTrainOneEpoch:
    """Tests for train_one_epoch."""

    def test_loss_finite(self):
        """One epoch of training produces a finite loss."""
        seed_everything(42)
        model = _make_model()
        loader = _make_loader(n=16, batch_size=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        device = torch.device("cpu")

        loss = train_one_epoch(
            model, loader, optimizer, criterion, scaler, device, lambda_tera=0.5,
        )
        assert np.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_handles_neg1_actions(self):
        """Training handles action=-1 (fainted mon) without NaN."""
        seed_everything(42)
        model = _make_model()
        # Use a dataset where every example has action_a = -1
        ds = TinyDataset(n=8)
        # Monkey-patch to make all action_a = -1
        original_getitem = ds.__getitem__

        def patched_getitem(idx):
            item = original_getitem(idx)
            item["action_a"] = -1
            return item

        ds.__getitem__ = patched_getitem
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        device = torch.device("cpu")

        loss = train_one_epoch(
            model, loader, optimizer, criterion, scaler, device, lambda_tera=0.5,
        )
        assert np.isfinite(loss), f"Loss should be finite with -1 actions, got {loss}"
        assert not np.isnan(loss), f"Loss should not be NaN with -1 actions"


# ===========================================================================
# TestValidate
# ===========================================================================


class TestValidate:
    """Tests for validate."""

    def test_returns_correct_shapes(self):
        """Collected outputs have shapes matching dataset size."""
        seed_everything(42)
        model = _make_model()
        n = 13  # not a multiple of batch_size
        loader = _make_loader(n=n, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        val_loss, collected = validate(
            model, loader, criterion, device, lambda_tera=0.5,
        )

        assert collected["logits_a"].shape == (n, 16)
        assert collected["logits_b"].shape == (n, 16)
        assert collected["logits_tera"].shape == (n, 3)
        assert collected["action_a"].shape == (n,)
        assert collected["action_b"].shape == (n,)
        assert collected["tera_label"].shape == (n,)
        assert collected["mask_a"].shape == (n, 16)
        assert collected["mask_b"].shape == (n, 16)

    def test_loss_finite(self):
        """Validation loss is finite."""
        seed_everything(42)
        model = _make_model()
        loader = _make_loader(n=16, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        val_loss, _ = validate(
            model, loader, criterion, device, lambda_tera=0.5,
        )
        assert np.isfinite(val_loss), f"Val loss should be finite, got {val_loss}"


# ===========================================================================
# TestConfigLoading
# ===========================================================================


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_bc_base(self):
        """Load configs/bc_base.yaml and verify expected keys."""
        config_path = Path(__file__).parent.parent / "configs" / "bc_base.yaml"
        config = load_config(config_path)

        assert config["model"]["d_model"] == 128
        assert config["model"]["n_layers"] == 4
        assert config["model"]["n_heads"] == 4
        assert config["model"]["d_ff"] == 512
        assert config["model"]["dropout"] == 0.1

        assert config["training"]["seed"] == 42
        assert config["training"]["batch_size"] == 512
        assert config["training"]["max_epochs"] == 50
        assert config["training"]["patience"] == 7
        assert config["training"]["lr"] == 3.0e-4
        assert config["training"]["weight_decay"] == 0.01
        assert config["training"]["label_smoothing"] == 0.02
        assert config["training"]["lambda_tera"] == 0.5
        assert config["training"]["require_both_actions"] is False
        assert config["training"]["num_workers"] == 4

        assert config["data"]["split_dir"] == "data/assembled"
        assert config["data"]["vocab_path"] is None

    def test_load_yaml_roundtrip(self):
        """Write a YAML config to temp file, load it back."""
        import yaml

        cfg = {
            "model": {"d_model": 32},
            "training": {"seed": 123, "lr": 1e-3},
            "data": {"split_dir": "/tmp/test"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name

        loaded = load_config(tmp_path)
        assert loaded["model"]["d_model"] == 32
        assert loaded["training"]["seed"] == 123
        assert loaded["data"]["split_dir"] == "/tmp/test"

        Path(tmp_path).unlink()
