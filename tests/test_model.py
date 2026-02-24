"""Tests for the Turn1Encoder and BCPolicy modules.

Covers: output shapes, lead embedding, field token, masking, valid
probabilities, shared action head, and checkpoint round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from turnone.models.encoder import EncoderConfig, Turn1Encoder
from turnone.models.bc_policy import BCPolicy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DUMMY_VOCAB = {"species": 50, "item": 30, "ability": 40, "tera_type": 20, "move": 80}
FAST_CFG = EncoderConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0)


def _random_inputs(B: int = 4, vocab: dict[str, int] = DUMMY_VOCAB):
    """Generate random model inputs for testing."""
    team_a = torch.stack(
        [torch.randint(0, vocab[k], (B, 6))
         for k in ["species", "item", "ability", "tera_type",
                    "move", "move", "move", "move"]],
        dim=-1,
    )
    team_b = torch.stack(
        [torch.randint(0, vocab[k], (B, 6))
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
    mask_a = torch.ones(B, 16, dtype=torch.bool)
    mask_b = torch.ones(B, 16, dtype=torch.bool)
    mask_a[:, 3] = False  # mask one slot
    mask_b[:, 7] = False
    return team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b


# ===========================================================================
# TestEncoder
# ===========================================================================


class TestEncoder:
    """Tests for Turn1Encoder."""

    def test_output_shape(self):
        """Random inputs produce output of shape (B, 13, d_model)."""
        B = 4
        encoder = Turn1Encoder(DUMMY_VOCAB, FAST_CFG)
        team_a, team_b, lead_a, lead_b, field_state, _, _ = _random_inputs(B)
        out = encoder(team_a, team_b, lead_a, lead_b, field_state)
        assert out.shape == (B, 13, FAST_CFG.d_model)

    def test_lead_embedding(self):
        """Verify lead positions get different lead embeddings than non-lead positions."""
        encoder = Turn1Encoder(DUMMY_VOCAB, FAST_CFG)

        # Build lead flags manually
        lead_a = torch.tensor([[0, 1]])  # (1, 2) -- mon 0 is lead A, mon 1 is lead B
        flags = Turn1Encoder._build_lead_flags(lead_a)  # (1, 6)

        assert flags[0, 0].item() == 1  # lead A
        assert flags[0, 1].item() == 2  # lead B
        assert flags[0, 2].item() == 0  # not a lead
        assert flags[0, 3].item() == 0
        assert flags[0, 4].item() == 0
        assert flags[0, 5].item() == 0

        # Verify the lead embeddings are distinct from non-lead
        emb_not_lead = encoder.emb_lead(torch.tensor(0))
        emb_lead_a = encoder.emb_lead(torch.tensor(1))
        emb_lead_b = encoder.emb_lead(torch.tensor(2))

        # With random init, these should almost surely be different
        assert not torch.allclose(emb_not_lead, emb_lead_a)
        assert not torch.allclose(emb_not_lead, emb_lead_b)
        assert not torch.allclose(emb_lead_a, emb_lead_b)

    def test_field_token_included(self):
        """Output has 13 tokens: 12 mons + 1 field."""
        B = 2
        encoder = Turn1Encoder(DUMMY_VOCAB, FAST_CFG)
        team_a, team_b, lead_a, lead_b, field_state, _, _ = _random_inputs(B)
        out = encoder(team_a, team_b, lead_a, lead_b, field_state)
        # 6 (team_a) + 6 (team_b) + 1 (field) = 13
        assert out.size(1) == 13


# ===========================================================================
# TestBCPolicy
# ===========================================================================


class TestBCPolicy:
    """Tests for BCPolicy."""

    def test_output_keys(self):
        """Forward returns a dict with the expected keys."""
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B=4)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)
        assert set(out.keys()) == {"logits_a", "logits_b", "logits_tera"}

    def test_output_shapes(self):
        """logits_a/b are (B, 16), logits_tera is (B, 3)."""
        B = 4
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B=B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)
        assert out["logits_a"].shape == (B, 16)
        assert out["logits_b"].shape == (B, 16)
        assert out["logits_tera"].shape == (B, 3)

    def test_masking(self):
        """Invalid slots have very large negative logits (effectively masked)."""
        B = 4
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B=B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

        # mask_a[:, 3] is False, so logits_a[:, 3] should be -30 (the mask fill value)
        assert torch.all(out["logits_a"][:, 3] == -30.0)
        # mask_b[:, 7] is False, so logits_b[:, 7] should be -30
        assert torch.all(out["logits_b"][:, 7] == -30.0)

        # Valid slots should have normal logit values (not the mask value)
        assert torch.all(out["logits_a"][:, 0] > -30.0)
        assert torch.all(out["logits_b"][:, 0] > -30.0)

    def test_valid_probs(self):
        """softmax(logits_a) sums to ~1 over valid slots."""
        B = 4
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)
        inputs = _random_inputs(B=B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

        probs_a = torch.softmax(out["logits_a"], dim=-1)
        probs_b = torch.softmax(out["logits_b"], dim=-1)

        # Total probability should sum to ~1
        assert torch.allclose(probs_a.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(probs_b.sum(dim=-1), torch.ones(B), atol=1e-5)

        # Masked slots should have 0 probability
        assert torch.allclose(probs_a[:, 3], torch.zeros(B), atol=1e-7)
        assert torch.allclose(probs_b[:, 7], torch.zeros(B), atol=1e-7)

    def test_shared_action_head(self):
        """Both leads use the same action_head weights."""
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)

        # The action head is a single nn.Sequential -- applied to both leads.
        # Verify by checking that identical inputs produce identical outputs.
        x = torch.randn(2, FAST_CFG.d_model)
        out1 = model.action_head(x)
        out2 = model.action_head(x)
        assert torch.allclose(out1, out2)

        # Also verify there is only ONE action head (not two separate ones)
        assert not hasattr(model, "action_head_a")
        assert not hasattr(model, "action_head_b")

    def test_checkpoint_roundtrip(self):
        """Save and reload produces the same outputs."""
        torch.manual_seed(42)
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        B = 2
        inputs = _random_inputs(B=B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs

        with torch.no_grad():
            out_orig = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

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
                },
                ckpt_path,
            )

        # Reload
        loaded = BCPolicy.from_checkpoint(ckpt_path, device=torch.device("cpu"))

        with torch.no_grad():
            out_loaded = loaded(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

        assert torch.allclose(out_orig["logits_a"], out_loaded["logits_a"], atol=1e-6)
        assert torch.allclose(out_orig["logits_b"], out_loaded["logits_b"], atol=1e-6)
        assert torch.allclose(out_orig["logits_tera"], out_loaded["logits_tera"], atol=1e-6)

        # Cleanup
        ckpt_path.unlink()


class TestAutoregressiveBC:
    """Tests for autoregressive BCPolicy."""

    def test_forward_conditioned_shape(self):
        """forward_conditioned returns correct output shapes."""
        torch.manual_seed(42)
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG, autoregressive=True)
        model.eval()

        B = 4
        inputs = _random_inputs(B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        action_a = torch.zeros(B, dtype=torch.long)

        with torch.no_grad():
            out = model.forward_conditioned(
                team_a, team_b, lead_a, lead_b, field_state,
                mask_a, mask_b, action_a,
            )

        assert out["logits_a"].shape == (B, 16)
        assert out["logits_b"].shape == (B, 16)
        assert out["logits_tera"].shape == (B, 3)

    def test_conditioned_differs_from_independent(self):
        """Conditioning on different action_a values should change logits_b."""
        torch.manual_seed(42)
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG, autoregressive=True)
        model.eval()

        B = 2
        inputs = _random_inputs(B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs

        with torch.no_grad():
            out0 = model.forward_conditioned(
                team_a, team_b, lead_a, lead_b, field_state,
                mask_a, mask_b, torch.zeros(B, dtype=torch.long),
            )
            out1 = model.forward_conditioned(
                team_a, team_b, lead_a, lead_b, field_state,
                mask_a, mask_b, torch.ones(B, dtype=torch.long),
            )

        # logits_a should be the same (not conditioned)
        assert torch.allclose(out0["logits_a"], out1["logits_a"], atol=1e-6)
        # logits_b should differ (conditioned on different action_a)
        assert not torch.allclose(out0["logits_b"], out1["logits_b"], atol=1e-6)

    def test_independent_forward_unchanged(self):
        """Standard forward() should work with autoregressive model."""
        torch.manual_seed(42)
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG, autoregressive=True)
        model.eval()

        B = 2
        inputs = _random_inputs(B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs

        with torch.no_grad():
            out = model(team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b)

        assert out["logits_a"].shape == (B, 16)
        assert out["logits_b"].shape == (B, 16)
        assert out["logits_tera"].shape == (B, 3)

    def test_autoregressive_checkpoint_roundtrip(self):
        """Save and reload autoregressive model produces same outputs."""
        torch.manual_seed(42)
        model = BCPolicy(DUMMY_VOCAB, FAST_CFG, autoregressive=True)
        model.eval()

        B = 2
        inputs = _random_inputs(B)
        team_a, team_b, lead_a, lead_b, field_state, mask_a, mask_b = inputs
        action_a = torch.zeros(B, dtype=torch.long)

        with torch.no_grad():
            out_orig = model.forward_conditioned(
                team_a, team_b, lead_a, lead_b, field_state,
                mask_a, mask_b, action_a,
            )

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
                    "autoregressive": True,
                },
                ckpt_path,
            )

        loaded = BCPolicy.from_checkpoint(ckpt_path, device=torch.device("cpu"))
        assert loaded.autoregressive is True

        with torch.no_grad():
            out_loaded = loaded.forward_conditioned(
                team_a, team_b, lead_a, lead_b, field_state,
                mask_a, mask_b, action_a,
            )

        assert torch.allclose(out_orig["logits_a"], out_loaded["logits_a"], atol=1e-6)
        assert torch.allclose(out_orig["logits_b"], out_loaded["logits_b"], atol=1e-6)
        assert torch.allclose(out_orig["logits_tera"], out_loaded["logits_tera"], atol=1e-6)

        ckpt_path.unlink()
