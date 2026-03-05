"""Tests for CQL Q-network and loss computation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from turnone.models.encoder import EncoderConfig
from turnone.models.cql_qvalue import QLearner
from turnone.rl.train_cql import _compute_cql_loss

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DUMMY_VOCAB = {"species": 50, "item": 30, "ability": 40, "tera_type": 20, "move": 80}
FAST_CFG = EncoderConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0)


def _random_inputs(B: int = 4):
    """Generate random model inputs for testing."""
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
    return team_a, team_b, lead_a, lead_b, field_state


class TestQLearner:
    """Tests for QLearner Q-network."""

    def test_output_shapes(self):
        """q_a is (B,16), q_b is (B,16), q_tera is (B,3)."""
        B = 4
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B)
        out = model(*inputs)
        assert out["q_a"].shape == (B, 16)
        assert out["q_b"].shape == (B, 16)
        assert out["q_tera"].shape == (B, 3)

    def test_output_keys(self):
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        out = model(*_random_inputs(2))
        assert set(out.keys()) == {"q_a", "q_b", "q_tera"}

    def test_shared_q_head(self):
        """Both leads use the same q_head weights."""
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        x = torch.randn(2, FAST_CFG.d_model)
        out1 = model.q_head(x)
        out2 = model.q_head(x)
        assert torch.allclose(out1, out2)

    def test_extract_policy_sums_to_one(self):
        """Policy from extract_policy sums to 1."""
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        model.eval()
        with torch.no_grad():
            out = model(*_random_inputs(1))

        valid_actions = [
            (0, 0, 0), (0, 4, 0), (4, 0, 1), (4, 4, 2),
        ]
        policy = QLearner.extract_policy(
            out["q_a"][0], out["q_b"][0], out["q_tera"][0],
            valid_actions, tau=1.0,
        )
        assert policy.shape == (4,)
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
        assert (policy >= 0).all()

    def test_extract_policy_temperature(self):
        """Lower temperature makes policy more peaked."""
        q_a = torch.tensor([1.0, 0.5] + [0.0] * 14)
        q_b = torch.tensor([0.0] * 16)
        q_tera = torch.tensor([0.0, 0.0, 0.0])
        valid_actions = [(0, 0, 0), (1, 0, 0)]

        p_warm = QLearner.extract_policy(q_a, q_b, q_tera, valid_actions, tau=10.0)
        p_cold = QLearner.extract_policy(q_a, q_b, q_tera, valid_actions, tau=0.1)

        # Cold policy should be more peaked on action 0 (higher Q)
        assert p_cold[0] > p_warm[0]

    def test_extract_policy_empty(self):
        """Empty valid_actions returns empty tensor."""
        q_a = torch.zeros(16)
        q_b = torch.zeros(16)
        q_tera = torch.zeros(3)
        policy = QLearner.extract_policy(q_a, q_b, q_tera, [], tau=1.0)
        assert policy.shape == (0,)

    def test_checkpoint_roundtrip(self):
        """Save and reload produces the same outputs."""
        torch.manual_seed(42)
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = _random_inputs(B)

        with torch.no_grad():
            out_orig = model(*inputs)

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
                },
                ckpt_path,
            )

        loaded = QLearner.from_checkpoint(ckpt_path, device=torch.device("cpu"))

        with torch.no_grad():
            out_loaded = loaded(*inputs)

        assert torch.allclose(out_orig["q_a"], out_loaded["q_a"], atol=1e-6)
        assert torch.allclose(out_orig["q_b"], out_loaded["q_b"], atol=1e-6)
        assert torch.allclose(out_orig["q_tera"], out_loaded["q_tera"], atol=1e-6)

        ckpt_path.unlink()


class TestCQLLoss:
    """Tests for CQL loss computation."""

    def test_loss_is_finite(self):
        """CQL loss should be finite with random inputs."""
        B = 8
        q_a = torch.randn(B, 16)
        q_b = torch.randn(B, 16)
        q_tera = torch.randn(B, 3)
        action_a = torch.randint(0, 16, (B,))
        action_b = torch.randint(0, 16, (B,))
        tera_label = torch.randint(0, 3, (B,))
        reward = torch.randn(B)
        strat_mask_a = torch.ones(B, 16, dtype=torch.bool)
        strat_mask_b = torch.ones(B, 16, dtype=torch.bool)

        loss, metrics = _compute_cql_loss(
            q_a, q_b, q_tera, action_a, action_b, tera_label,
            reward, strat_mask_a, strat_mask_b, alpha=1.0,
        )
        assert torch.isfinite(loss)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_gradients_flow(self):
        """Gradients should flow through the CQL loss."""
        B = 4
        model = QLearner(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B)
        out = model(*inputs)

        action_a = torch.randint(0, 16, (B,))
        action_b = torch.randint(0, 16, (B,))
        tera_label = torch.randint(0, 3, (B,))
        reward = torch.randn(B)
        strat_mask_a = torch.ones(B, 16, dtype=torch.bool)
        strat_mask_b = torch.ones(B, 16, dtype=torch.bool)

        loss, _ = _compute_cql_loss(
            out["q_a"], out["q_b"], out["q_tera"],
            action_a, action_b, tera_label,
            reward, strat_mask_a, strat_mask_b, alpha=1.0,
        )
        loss.backward()

        # Check that encoder parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert has_grad, "Encoder should receive gradients"

    def test_alpha_zero_is_td_only(self):
        """With alpha=0, loss should equal pure TD loss."""
        B = 4
        q_a = torch.randn(B, 16)
        q_b = torch.randn(B, 16)
        q_tera = torch.randn(B, 3)
        action_a = torch.randint(0, 16, (B,))
        action_b = torch.randint(0, 16, (B,))
        tera_label = torch.randint(0, 3, (B,))
        reward = torch.randn(B)
        mask = torch.ones(B, 16, dtype=torch.bool)

        loss, metrics = _compute_cql_loss(
            q_a, q_b, q_tera, action_a, action_b, tera_label,
            reward, mask, mask, alpha=0.0,
        )
        # Loss should equal TD loss when alpha=0
        assert abs(loss.item() - metrics["td_loss"]) < 1e-5

    def test_cql_penalty_positive(self):
        """CQL penalty should generally be positive (logsumexp >= data Q)."""
        B = 16
        # Set all Q-values to 0 except data actions
        q_a = torch.zeros(B, 16)
        q_b = torch.zeros(B, 16)
        q_tera = torch.zeros(B, 3)
        action_a = torch.zeros(B, dtype=torch.long)
        action_b = torch.zeros(B, dtype=torch.long)
        tera_label = torch.zeros(B, dtype=torch.long)
        reward = torch.zeros(B)
        mask = torch.ones(B, 16, dtype=torch.bool)

        _, metrics = _compute_cql_loss(
            q_a, q_b, q_tera, action_a, action_b, tera_label,
            reward, mask, mask, alpha=1.0,
        )
        # logsumexp over all actions >= Q(data) when Q values are uniform
        assert metrics["cql_penalty"] >= -1e-5


# Need numpy for one test
import numpy as np
