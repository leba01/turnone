"""Tests for the DynamicsModel and dynamics training loop.

Covers: output shapes, action embedding sensitivity, no-action embedding,
gradient flow, checkpoint round-trip, parameter count, field decomposition,
loss computation, one-epoch training, and config loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from turnone.models.encoder import EncoderConfig
from turnone.models.dynamics import DynamicsModel, FieldLogits, remap_actions, NO_ACTION_IDX

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DUMMY_VOCAB = {"species": 50, "item": 30, "ability": 40, "tera_type": 20, "move": 80}
FAST_CFG = EncoderConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0)


def _random_dynamics_inputs(B: int = 4):
    """Generate random inputs for DynamicsModel."""
    team_a = torch.stack(
        [torch.randint(0, DUMMY_VOCAB[k], (B, 6))
         for k in ["species", "item", "ability", "tera_type",
                    "move", "move", "move", "move"]],
        dim=-1,
    )
    team_b = torch.stack(
        [torch.randint(0, DUMMY_VOCAB[k], (B, 6))
         for k in ["species", "item", "ability", "tera_type",
                    "move", "move", "move", "move"]],
        dim=-1,
    )
    lead_a = torch.stack(
        [torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)], dim=-1
    )
    lead_b = torch.stack(
        [torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)], dim=-1
    )
    field_state = torch.randn(B, 5)
    action_a = torch.randint(0, 16, (B,))
    action_b = torch.randint(0, 16, (B,))
    tera_flag = torch.randint(0, 3, (B,))
    opp_action_a = torch.randint(0, 16, (B,))
    opp_action_b = torch.randint(0, 16, (B,))
    opp_tera_flag = torch.randint(0, 3, (B,))
    return (team_a, team_b, lead_a, lead_b, field_state,
            action_a, action_b, tera_flag,
            opp_action_a, opp_action_b, opp_tera_flag)


# ===========================================================================
# TestDynamicsModel
# ===========================================================================


class TestDynamicsModel:
    """Tests for DynamicsModel."""

    def test_output_shapes(self):
        """hp_pred (B,4), ko_logits (B,4), field_logits decomposed."""
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_dynamics_inputs(B=B)
        hp_pred, ko_logits, field_logits = model(*inputs)
        assert hp_pred.shape == (B, 4)
        assert ko_logits.shape == (B, 4)
        assert isinstance(field_logits, FieldLogits)
        assert field_logits.weather.shape == (B, 5)
        assert field_logits.terrain.shape == (B, 5)
        assert field_logits.binary.shape == (B, 3)

    def test_predict_field_state_shape(self):
        """predict_field_state returns (B, 5) from FieldLogits."""
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_dynamics_inputs(B=B)
        _, _, field_logits = model(*inputs)
        field_pred = model.predict_field_state(field_logits)
        assert field_pred.shape == (B, 5)

    def test_action_embedding(self):
        """Different actions produce different outputs."""
        torch.manual_seed(123)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = list(_random_dynamics_inputs(B=B))

        # Run with action_a = 0 for all
        inputs_0 = list(inputs)
        inputs_0[5] = torch.zeros(B, dtype=torch.long)
        with torch.no_grad():
            hp_0, ko_0, _ = model(*inputs_0)

        # Run with action_a = 5 for all
        inputs_5 = list(inputs)
        inputs_5[5] = torch.full((B,), 5, dtype=torch.long)
        with torch.no_grad():
            hp_5, ko_5, _ = model(*inputs_5)

        assert not torch.allclose(hp_0, hp_5, atol=1e-6)
        assert not torch.allclose(ko_0, ko_5, atol=1e-6)

    def test_no_action_embedding(self):
        """Index 16 (no-action/fainted) works and differs from index 0."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = list(_random_dynamics_inputs(B=B))

        # Run with action_a = 0
        inputs_0 = list(inputs)
        inputs_0[5] = torch.zeros(B, dtype=torch.long)
        with torch.no_grad():
            hp_0, _, _ = model(*inputs_0)

        # Run with action_a = 16 (no-action)
        inputs_16 = list(inputs)
        inputs_16[5] = torch.full((B,), NO_ACTION_IDX, dtype=torch.long)
        with torch.no_grad():
            hp_16, _, _ = model(*inputs_16)

        assert not torch.allclose(hp_0, hp_16, atol=1e-6)

    def test_remap_actions(self):
        """remap_actions maps -1 -> 16, leaves 0-15 unchanged."""
        actions = torch.tensor([0, 5, -1, 15, -1])
        remapped = remap_actions(actions)
        assert remapped.tolist() == [0, 5, NO_ACTION_IDX, 15, NO_ACTION_IDX]

    def test_gradients_flow(self):
        """All parameters have gradients after backward pass."""
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_dynamics_inputs(B=4)
        hp_pred, ko_logits, field_logits = model(*inputs)

        # Combine losses
        loss = (hp_pred.sum() + ko_logits.sum()
                + field_logits.weather.sum() + field_logits.terrain.sum()
                + field_logits.binary.sum())
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_checkpoint_roundtrip(self):
        """Save and reload produces the same outputs."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = _random_dynamics_inputs(B=B)

        with torch.no_grad():
            hp_orig, ko_orig, fl_orig = model(*inputs)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
            torch.save(
                {
                    "encoder_config": {
                        "d_model": FAST_CFG.d_model,
                        "n_layers": FAST_CFG.n_layers,
                        "n_heads": FAST_CFG.n_heads,
                        "d_ff": FAST_CFG.d_ff,
                        "dropout": FAST_CFG.dropout,
                    },
                    "vocab_sizes": DUMMY_VOCAB,
                    "model_state_dict": model.state_dict(),
                    "d_action": model.d_action,
                    "d_hidden": model.d_hidden,
                    "n_mlp_layers": model.n_mlp_layers,
                    "dropout": model._dropout,
                },
                ckpt_path,
            )

        # Reload
        loaded = DynamicsModel.from_checkpoint(ckpt_path, device=torch.device("cpu"))

        with torch.no_grad():
            hp_loaded, ko_loaded, fl_loaded = loaded(*inputs)

        assert torch.allclose(hp_orig, hp_loaded, atol=1e-6)
        assert torch.allclose(ko_orig, ko_loaded, atol=1e-6)
        assert torch.allclose(fl_orig.weather, fl_loaded.weather, atol=1e-6)
        assert torch.allclose(fl_orig.terrain, fl_loaded.terrain, atol=1e-6)
        assert torch.allclose(fl_orig.binary, fl_loaded.binary, atol=1e-6)

        # Cleanup
        ckpt_path.unlink()

    def test_param_count(self):
        """Model has reasonable number of parameters (> 100K)."""
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 100_000, f"Too few parameters: {n_params:,}"

    def test_encode_state_shape(self):
        """encode_state returns (B, d_model) pooled representation."""
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()
        inputs = _random_dynamics_inputs(B=B)
        with torch.no_grad():
            pooled = model.encode_state(*inputs[:5])
        assert pooled.shape == (B, FAST_CFG.d_model)

    def test_predict_from_pooled_matches_forward(self):
        """predict_from_pooled gives identical results to forward."""
        torch.manual_seed(99)
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()
        inputs = _random_dynamics_inputs(B=B)

        with torch.no_grad():
            hp_fwd, ko_fwd, fl_fwd = model(*inputs)
            pooled = model.encode_state(*inputs[:5])
            hp_split, ko_split, fl_split = model.predict_from_pooled(
                pooled, *inputs[5:],
            )

        assert torch.allclose(hp_fwd, hp_split, atol=1e-6)
        assert torch.allclose(ko_fwd, ko_split, atol=1e-6)
        assert torch.allclose(fl_fwd.weather, fl_split.weather, atol=1e-6)
        assert torch.allclose(fl_fwd.terrain, fl_split.terrain, atol=1e-6)
        assert torch.allclose(fl_fwd.binary, fl_split.binary, atol=1e-6)

    def test_encode_state_reuse(self):
        """Encoding once and reusing for different actions gives consistent results."""
        torch.manual_seed(77)
        B = 2
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()
        inputs = _random_dynamics_inputs(B=B)

        with torch.no_grad():
            pooled = model.encode_state(*inputs[:5])

            # Different action_a values
            actions1 = list(inputs[5:])
            actions1[0] = torch.zeros(B, dtype=torch.long)
            hp_1, _, _ = model.predict_from_pooled(pooled, *actions1)

            actions2 = list(inputs[5:])
            actions2[0] = torch.full((B,), 5, dtype=torch.long)
            hp_2, _, _ = model.predict_from_pooled(pooled, *actions2)

        # Different actions should produce different outputs
        assert not torch.allclose(hp_1, hp_2, atol=1e-6)


# ===========================================================================
# TestTrainDynamics
# ===========================================================================


class TestTrainDynamics:
    """Tests for dynamics training infrastructure."""

    def test_loss_computation(self):
        """Compute decomposed loss on random batch, verify finite."""
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_dynamics_inputs(B=4)
        hp_pred, ko_logits, field_logits = model(*inputs)

        B = 4
        hp_delta = torch.randn(B, 4)
        ko_flags = torch.randint(0, 2, (B, 4)).float()
        weather_target = torch.randint(0, 5, (B,))
        terrain_target = torch.randint(0, 5, (B,))
        binary_target = torch.randint(0, 2, (B, 3)).float()

        hp_loss = nn.MSELoss()(hp_pred, hp_delta)
        ko_loss = nn.BCEWithLogitsLoss()(ko_logits, ko_flags)
        weather_loss = nn.CrossEntropyLoss()(field_logits.weather, weather_target)
        terrain_loss = nn.CrossEntropyLoss()(field_logits.terrain, terrain_target)
        binary_loss = nn.BCEWithLogitsLoss()(field_logits.binary, binary_target)

        total_loss = hp_loss + 3.0 * ko_loss + 0.5 * (weather_loss + terrain_loss + binary_loss)

        assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
        assert total_loss.item() > 0, "Loss should be positive"

    def test_train_one_epoch(self):
        """Run one training step on random data, verify loss is finite."""
        from turnone.models.train_dynamics import train_one_epoch

        torch.manual_seed(0)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        device = torch.device("cpu")

        hp_criterion = nn.MSELoss()
        ko_criterion = nn.BCEWithLogitsLoss()
        weather_criterion = nn.CrossEntropyLoss()
        terrain_criterion = nn.CrossEntropyLoss()
        binary_field_criterion = nn.BCEWithLogitsLoss()

        # Build a tiny DataLoader with random data
        B = 8
        dataset = []
        for _ in range(3):
            inputs = _random_dynamics_inputs(B=B)
            batch = {
                "team_a": inputs[0],
                "team_b": inputs[1],
                "lead_a": inputs[2],
                "lead_b": inputs[3],
                "field_state": inputs[4],
                "action_a": inputs[5],
                "action_b": inputs[6],
                "tera_label": inputs[7],
                "opp_action_a": inputs[8],
                "opp_action_b": inputs[9],
                "opp_tera_label": inputs[10],
                "hp_delta": torch.randn(B, 4),
                "ko_flags": torch.randint(0, 2, (B, 4)).float(),
                # field_after: [weather(0-4), terrain(0-4), trick_room, tw_p1, tw_p2]
                "field_after": torch.cat([
                    torch.randint(0, 5, (B, 1)).float(),
                    torch.randint(0, 5, (B, 1)).float(),
                    torch.randint(0, 2, (B, 3)).float(),
                ], dim=-1),
            }
            dataset.append(batch)

        avg_loss = train_one_epoch(
            model, dataset, optimizer,
            hp_criterion, ko_criterion,
            weather_criterion, terrain_criterion, binary_field_criterion,
            scaler, device,
            lambda_ko=3.0, lambda_field=0.5,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss > 0, "Average loss should be positive"
        assert not (avg_loss != avg_loss), "Loss is NaN"

    def test_config_loading_v2(self):
        """Load dynamics_v2.yaml and verify new keys."""
        from turnone.models.train_dynamics import load_config

        config_path = Path(__file__).parent.parent / "configs" / "dynamics_v2.yaml"
        config = load_config(config_path)

        model_cfg = config["model"]
        assert model_cfg["d_action"] == 64
        assert model_cfg["d_hidden"] == 512
        assert model_cfg["n_mlp_layers"] == 4
        assert model_cfg["action_cross_attn"] is True
        assert model_cfg["action_attn_heads"] == 2
        assert model_cfg["action_attn_layers"] == 1

    def test_config_loading(self):
        """Load dynamics_base.yaml and verify keys."""
        from turnone.models.train_dynamics import load_config

        config_path = Path(__file__).parent.parent / "configs" / "dynamics_base.yaml"
        config = load_config(config_path)

        assert "model" in config
        assert "training" in config
        assert "data" in config

        model_cfg = config["model"]
        assert model_cfg["d_model"] == 128
        assert model_cfg["n_layers"] == 4
        assert model_cfg["n_heads"] == 4
        assert model_cfg["d_ff"] == 512
        assert model_cfg["d_action"] == 32
        assert model_cfg["d_hidden"] == 256
        assert model_cfg["n_mlp_layers"] == 3

        train_cfg = config["training"]
        assert train_cfg["seed"] == 42
        assert train_cfg["batch_size"] == 512
        assert train_cfg["lambda_ko"] == 3.0
        assert train_cfg["lambda_field"] == 0.5
        assert train_cfg["require_both_actions"] is True
        assert train_cfg["patience"] == 10


# ===========================================================================
# TestCrossAttention
# ===========================================================================


class TestCrossAttention:
    """Tests for action cross-attention (dynamics v2)."""

    def test_v2_output_shapes(self):
        """v2 model produces same output shapes as v1."""
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG, action_cross_attn=True)
        inputs = _random_dynamics_inputs(B=B)
        hp_pred, ko_logits, field_logits = model(*inputs)
        assert hp_pred.shape == (B, 4)
        assert ko_logits.shape == (B, 4)
        assert field_logits.weather.shape == (B, 5)
        assert field_logits.terrain.shape == (B, 5)
        assert field_logits.binary.shape == (B, 3)

    def test_v1_default_no_cross_attn(self):
        """Default model has no action_attn attribute."""
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        assert model.action_cross_attn is False
        assert not hasattr(model, "action_attn")
        assert not hasattr(model, "action_pos_emb")

    def test_v2_gradients_flow(self):
        """Gradients reach action_pos_emb and action_attn."""
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG, action_cross_attn=True)
        inputs = _random_dynamics_inputs(B=4)
        hp_pred, ko_logits, field_logits = model(*inputs)

        loss = (hp_pred.sum() + ko_logits.sum()
                + field_logits.weather.sum() + field_logits.terrain.sum()
                + field_logits.binary.sum())
        loss.backward()

        # Check cross-attention specific params
        assert model.action_pos_emb.weight.grad is not None
        assert torch.isfinite(model.action_pos_emb.weight.grad).all()
        for name, param in model.action_attn.named_parameters():
            assert param.grad is not None, f"No gradient for action_attn.{name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for action_attn.{name}"

    def test_v2_checkpoint_roundtrip(self):
        """Save and reload v2 model produces identical outputs."""
        torch.manual_seed(42)
        model = DynamicsModel(
            DUMMY_VOCAB, FAST_CFG, d_action=16,
            action_cross_attn=True, action_attn_heads=2,
        )
        model.eval()

        B = 2
        inputs = _random_dynamics_inputs(B=B)
        with torch.no_grad():
            hp_orig, ko_orig, fl_orig = model(*inputs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
            torch.save(
                {
                    "encoder_config": {
                        "d_model": FAST_CFG.d_model,
                        "n_layers": FAST_CFG.n_layers,
                        "n_heads": FAST_CFG.n_heads,
                        "d_ff": FAST_CFG.d_ff,
                        "dropout": FAST_CFG.dropout,
                    },
                    "vocab_sizes": DUMMY_VOCAB,
                    "model_state_dict": model.state_dict(),
                    "d_action": model.d_action,
                    "d_hidden": model.d_hidden,
                    "n_mlp_layers": model.n_mlp_layers,
                    "dropout": model._dropout,
                    "action_cross_attn": model.action_cross_attn,
                    "action_attn_heads": model.action_attn_heads,
                    "action_attn_layers": model.action_attn_layers,
                },
                ckpt_path,
            )

        loaded = DynamicsModel.from_checkpoint(ckpt_path, device=torch.device("cpu"))
        assert loaded.action_cross_attn is True

        with torch.no_grad():
            hp_loaded, ko_loaded, fl_loaded = loaded(*inputs)

        assert torch.allclose(hp_orig, hp_loaded, atol=1e-6)
        assert torch.allclose(ko_orig, ko_loaded, atol=1e-6)
        assert torch.allclose(fl_orig.weather, fl_loaded.weather, atol=1e-6)

        ckpt_path.unlink()

    def test_v1_checkpoint_loads_without_cross_attn_keys(self):
        """v1 checkpoint (no cross-attn keys) loads fine via from_checkpoint."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = _random_dynamics_inputs(B=B)
        with torch.no_grad():
            hp_orig, _, _ = model(*inputs)

        # Save v1-style checkpoint (no cross-attn keys)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = Path(f.name)
            torch.save(
                {
                    "encoder_config": {
                        "d_model": FAST_CFG.d_model,
                        "n_layers": FAST_CFG.n_layers,
                        "n_heads": FAST_CFG.n_heads,
                        "d_ff": FAST_CFG.d_ff,
                        "dropout": FAST_CFG.dropout,
                    },
                    "vocab_sizes": DUMMY_VOCAB,
                    "model_state_dict": model.state_dict(),
                    "d_action": model.d_action,
                    "d_hidden": model.d_hidden,
                    "n_mlp_layers": model.n_mlp_layers,
                    "dropout": model._dropout,
                },
                ckpt_path,
            )

        loaded = DynamicsModel.from_checkpoint(ckpt_path, device=torch.device("cpu"))
        assert loaded.action_cross_attn is False

        with torch.no_grad():
            hp_loaded, _, _ = loaded(*inputs)

        assert torch.allclose(hp_orig, hp_loaded, atol=1e-6)
        ckpt_path.unlink()

    def test_v2_predict_from_pooled_matches_forward(self):
        """Two-stage API still works with cross-attention."""
        torch.manual_seed(99)
        B = 4
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG, action_cross_attn=True)
        model.eval()
        inputs = _random_dynamics_inputs(B=B)

        with torch.no_grad():
            hp_fwd, ko_fwd, fl_fwd = model(*inputs)
            pooled = model.encode_state(*inputs[:5])
            hp_split, ko_split, fl_split = model.predict_from_pooled(
                pooled, *inputs[5:],
            )

        assert torch.allclose(hp_fwd, hp_split, atol=1e-6)
        assert torch.allclose(ko_fwd, ko_split, atol=1e-6)
        assert torch.allclose(fl_fwd.weather, fl_split.weather, atol=1e-6)

    def test_v2_param_count(self):
        """v2 has more parameters than v1."""
        v1 = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        v2 = DynamicsModel(DUMMY_VOCAB, FAST_CFG, action_cross_attn=True)
        n_v1 = sum(p.numel() for p in v1.parameters())
        n_v2 = sum(p.numel() for p in v2.parameters())
        assert n_v2 > n_v1, f"v2 ({n_v2:,}) should have more params than v1 ({n_v1:,})"
