"""Tests for payoff matrix construction and evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from turnone.data.action_space import compute_action_mask, compute_strategic_mask
from turnone.models.encoder import EncoderConfig
from turnone.models.dynamics import DynamicsModel, FieldLogits
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix, _compute_reward_gpu
from turnone.eval.dynamics_metrics import compute_dynamics_metrics
from turnone.rl.reward import compute_reward_from_dynamics

DUMMY_VOCAB = {"species": 50, "item": 30, "ability": 40, "tera_type": 20, "move": 80}
FAST_CFG = EncoderConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0)

_FIELD_ORDER = ["species", "item", "ability", "tera_type", "move", "move", "move", "move"]


def _random_team() -> torch.Tensor:
    """Generate a single random team (6, 8) with per-column vocab bounds."""
    cols = [torch.randint(0, DUMMY_VOCAB[k], (6,)) for k in _FIELD_ORDER]
    return torch.stack(cols, dim=-1)


class TestEnumerateJointActions:
    """Tests for enumerate_joint_actions."""

    def test_full_masks(self):
        """All slots valid -> 16 * 16 * 3 = 768 actions."""
        mask_a = np.ones(16, dtype=bool)
        mask_b = np.ones(16, dtype=bool)
        actions = enumerate_joint_actions(mask_a, mask_b, include_tera=True)
        assert len(actions) == 16 * 16 * 3

    def test_no_tera(self):
        """Without tera -> 16 * 16 = 256 actions."""
        mask_a = np.ones(16, dtype=bool)
        mask_b = np.ones(16, dtype=bool)
        actions = enumerate_joint_actions(mask_a, mask_b, include_tera=False)
        assert len(actions) == 16 * 16

    def test_partial_masks(self):
        """Only some slots valid."""
        mask_a = np.zeros(16, dtype=bool)
        mask_a[[0, 4, 8]] = True  # 3 valid slots
        mask_b = np.zeros(16, dtype=bool)
        mask_b[[0, 1]] = True  # 2 valid slots
        actions = enumerate_joint_actions(mask_a, mask_b, include_tera=True)
        assert len(actions) == 3 * 2 * 3  # 18

    def test_tuples_valid(self):
        """All tuples have valid ranges."""
        mask_a = np.ones(16, dtype=bool)
        mask_b = np.ones(16, dtype=bool)
        actions = enumerate_joint_actions(mask_a, mask_b)
        for sa, sb, tf in actions:
            assert 0 <= sa < 16
            assert 0 <= sb < 16
            assert 0 <= tf < 3


class TestBuildPayoffMatrix:
    """Tests for build_payoff_matrix."""

    def test_shape(self):
        """Payoff matrix has correct shape."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        state = {
            "team_a": _random_team(),
            "team_b": _random_team(),
            "lead_a": torch.tensor([0, 1]),
            "lead_b": torch.tensor([0, 1]),
            "field_state": torch.zeros(5),
        }
        field_before = np.zeros(5, dtype=np.float32)

        mask_a = np.zeros(16, dtype=bool)
        mask_a[[0, 3]] = True
        mask_b = np.zeros(16, dtype=bool)
        mask_b[[0, 3]] = True

        actions_p1 = enumerate_joint_actions(mask_a, mask_b, include_tera=False)
        actions_p2 = enumerate_joint_actions(mask_a, mask_b, include_tera=False)

        R = build_payoff_matrix(
            model, state, actions_p1, actions_p2,
            field_before, torch.device("cpu"),
        )
        assert R.shape == (len(actions_p1), len(actions_p2))

    def test_finite_values(self):
        """All payoff matrix entries should be finite."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        state = {
            "team_a": _random_team(),
            "team_b": _random_team(),
            "lead_a": torch.tensor([0, 1]),
            "lead_b": torch.tensor([0, 1]),
            "field_state": torch.zeros(5),
        }
        field_before = np.zeros(5, dtype=np.float32)

        mask_a = np.zeros(16, dtype=bool)
        mask_a[0] = True
        mask_b = np.zeros(16, dtype=bool)
        mask_b[0] = True

        actions = enumerate_joint_actions(mask_a, mask_b, include_tera=False)

        R = build_payoff_matrix(
            model, state, actions, actions,
            field_before, torch.device("cpu"),
        )
        assert np.all(np.isfinite(R))


class TestGPUReward:
    """Tests that GPU reward computation matches CPU numpy version."""

    def test_gpu_reward_matches_cpu(self):
        """_compute_reward_gpu matches compute_reward_from_dynamics."""
        rng = np.random.RandomState(42)
        N = 64
        hp_pred_np = rng.randn(N, 4).astype(np.float32) * 20
        ko_logits_np = rng.randn(N, 4).astype(np.float32) * 2
        field_pred_np = np.zeros((N, 5), dtype=np.float32)
        field_pred_np[:, 3] = rng.rand(N)  # tailwind_p1
        field_pred_np[:, 4] = rng.rand(N)  # tailwind_p2
        field_pred_np[:, 2] = (rng.rand(N) > 0.5).astype(np.float32)  # trick room
        field_before_np = np.zeros((N, 5), dtype=np.float32)
        field_before_np[:, 3] = rng.rand(N)
        field_before_np[:, 4] = rng.rand(N)

        # CPU version
        rewards_cpu = compute_reward_from_dynamics(
            hp_pred_np, ko_logits_np, field_pred_np, field_before_np,
        )

        # GPU version
        rewards_gpu = _compute_reward_gpu(
            torch.from_numpy(hp_pred_np),
            torch.from_numpy(ko_logits_np),
            torch.from_numpy(field_pred_np),
            torch.from_numpy(field_before_np),
        ).numpy()

        np.testing.assert_allclose(rewards_cpu, rewards_gpu, atol=1e-5)

    def test_payoff_uses_cached_encoder(self):
        """build_payoff_matrix with small action space produces correct shape and finite values."""
        torch.manual_seed(42)
        model = DynamicsModel(DUMMY_VOCAB, FAST_CFG)
        model.eval()

        state = {
            "team_a": _random_team(),
            "team_b": _random_team(),
            "lead_a": torch.tensor([0, 1]),
            "lead_b": torch.tensor([0, 1]),
            "field_state": torch.zeros(5),
        }
        field_before = np.zeros(5, dtype=np.float32)

        mask_a = np.zeros(16, dtype=bool)
        mask_a[[0, 1, 2]] = True  # 3 valid slots
        mask_b = np.zeros(16, dtype=bool)
        mask_b[[0, 1]] = True  # 2 valid slots

        actions_p1 = enumerate_joint_actions(mask_a, mask_b, include_tera=True)
        actions_p2 = enumerate_joint_actions(mask_a, mask_b, include_tera=True)

        # 3*2*3 = 18 actions per side -> 18x18 matrix
        assert len(actions_p1) == 18

        R = build_payoff_matrix(
            model, state, actions_p1, actions_p2,
            field_before, torch.device("cpu"),
            batch_size=32,  # small batch to test batching
        )
        assert R.shape == (18, 18)
        assert np.all(np.isfinite(R))


class TestDynamicsMetrics:
    """Tests for dynamics_metrics."""

    def test_perfect_predictions(self):
        """Perfect predictions give MAE=0, R²=1, accuracy=1."""
        N = 100
        hp = np.random.RandomState(42).randn(N, 4).astype(np.float32) * 10
        ko = np.zeros((N, 4), dtype=np.float32)
        ko[:10, 0] = 1.0  # some KOs
        field = np.zeros((N, 5), dtype=np.float32)
        field[:, 0] = 1.0  # weather = 1
        field[:, 3] = 1.0  # tailwind

        # Perfect predictions
        ko_logits = np.where(ko > 0.5, 30.0, -30.0).astype(np.float32)

        metrics = compute_dynamics_metrics(hp, hp, ko_logits, ko, field, field)
        assert metrics["hp_mae"] < 1e-6
        assert metrics["hp_r2"] > 0.99
        assert metrics["ko_acc"] > 0.99
        assert metrics["weather_acc"] > 0.99
        assert metrics["binary_field_acc"] > 0.99

    def test_output_keys(self):
        """All expected metric keys present."""
        N = 10
        hp = np.zeros((N, 4), dtype=np.float32)
        ko = np.zeros((N, 4), dtype=np.float32)
        ko_logits = np.zeros((N, 4), dtype=np.float32)
        field = np.zeros((N, 5), dtype=np.float32)

        metrics = compute_dynamics_metrics(hp, hp, ko_logits, ko, field, field)
        expected = {"hp_mae", "hp_rmse", "hp_r2", "ko_auc", "ko_acc",
                    "ko_bce", "weather_acc", "terrain_acc", "binary_field_acc"}
        assert set(metrics.keys()) == expected


class TestStrategicMask:
    """Tests for compute_strategic_mask."""

    def test_excludes_self_target_for_single(self):
        """Strategic mask should NOT include target=3 or target=2 for single-target moves."""
        # Thunderbolt = single-target, Protect = self-target
        moves = ["Thunderbolt", "Protect", "UNK", "UNK"]
        bc_mask = compute_action_mask(moves)
        strat_mask = compute_strategic_mask(moves)

        # Thunderbolt is move 0 -> slots 0,1,2,3
        # BC mask allows target=3 (slot 3)
        assert bc_mask[3] == True  # noqa: E712
        # Strategic mask should NOT allow target=3 for single-target
        assert strat_mask[3] == False  # noqa: E712
        # Strategic mask should NOT allow target=2 (ally) for normal single-target
        assert strat_mask[2] == False  # noqa: E712

        # Opponents still valid
        assert strat_mask[0] == True  # noqa: E712
        assert strat_mask[1] == True  # noqa: E712

    def test_keeps_self_target_for_self_moves(self):
        """Strategic mask should keep target=3 for self-target moves (Protect etc)."""
        moves = ["Protect", "Swords Dance", "UNK", "UNK"]
        strat_mask = compute_strategic_mask(moves)

        # Protect is move 0 -> slot 3 (target=3) should be valid
        assert strat_mask[3] == True  # noqa: E712
        # Swords Dance is move 1 -> slot 7 (target=3) should be valid
        assert strat_mask[7] == True  # noqa: E712

    def test_keeps_self_target_for_spread_moves(self):
        """Strategic mask should keep target=3 for spread moves (Earthquake etc)."""
        moves = ["Earthquake", "Heat Wave", "UNK", "UNK"]
        strat_mask = compute_strategic_mask(moves)

        # Earthquake is move 0 -> slot 3 (target=3) should be valid
        assert strat_mask[3] == True  # noqa: E712
        # Heat Wave is move 1 -> slot 7 (target=3) should be valid
        assert strat_mask[7] == True  # noqa: E712

    def test_allows_ally_for_dual_purpose(self):
        """Strategic mask allows ally targeting for Pollen Puff etc."""
        moves = ["Pollen Puff", "Thunderbolt", "UNK", "UNK"]
        strat_mask = compute_strategic_mask(moves)

        # Pollen Puff (move 0): target=2 (ally) should be valid
        assert strat_mask[2] == True   # noqa: E712 — ally
        assert strat_mask[0] == True   # noqa: E712 — opp A
        assert strat_mask[1] == True   # noqa: E712 — opp B
        assert strat_mask[3] == False  # noqa: E712 — self (still excluded)

        # Thunderbolt (move 1): target=2 should NOT be valid
        assert strat_mask[6] == False  # noqa: E712 — slot 1*4+2 = ally
        assert strat_mask[4] == True   # noqa: E712 — opp A
        assert strat_mask[5] == True   # noqa: E712 — opp B

    def test_strategic_subset_of_bc(self):
        """Strategic mask should be a subset of BC mask for any moveset."""
        moves = ["Thunderbolt", "Protect", "Earthquake", "Ice Beam"]
        bc_mask = compute_action_mask(moves)
        strat_mask = compute_strategic_mask(moves)

        # Every slot valid in strategic should also be valid in BC
        assert np.all(strat_mask <= bc_mask)


class TestCanonicalizeTargets:
    """Tests for _canonicalize_action_slot."""

    def test_single_target_remapped(self):
        """target=3 for single-target move should be remapped to target=0."""
        from turnone.data.dataset import _canonicalize_action_slot
        # Thunderbolt is move 0, target=3 => slot 3
        team = [{"moves": ["Thunderbolt", "Protect", "UNK", "UNK"],
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] + [{"moves": ["UNK"]*4,
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] * 5
        result = _canonicalize_action_slot(3, team, 0)
        # Should remap to move_idx=0, target=0 => slot 0
        assert result == 0

    def test_self_target_preserved(self):
        """target=3 for self-target move should NOT be remapped."""
        from turnone.data.dataset import _canonicalize_action_slot
        team = [{"moves": ["Protect", "UNK", "UNK", "UNK"],
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] + [{"moves": ["UNK"]*4,
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] * 5
        result = _canonicalize_action_slot(3, team, 0)
        assert result == 3  # unchanged

    def test_spread_target_preserved(self):
        """target=3 for spread move should NOT be remapped."""
        from turnone.data.dataset import _canonicalize_action_slot
        team = [{"moves": ["Earthquake", "UNK", "UNK", "UNK"],
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] + [{"moves": ["UNK"]*4,
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] * 5
        result = _canonicalize_action_slot(3, team, 0)
        assert result == 3  # unchanged

    def test_fainted_preserved(self):
        """Fainted slot (-1) should be unchanged."""
        from turnone.data.dataset import _canonicalize_action_slot
        team = [{"moves": ["Thunderbolt", "UNK", "UNK", "UNK"],
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] + [{"moves": ["UNK"]*4,
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] * 5
        result = _canonicalize_action_slot(-1, team, 0)
        assert result == -1

    def test_non_self_target_unchanged(self):
        """target=0 (opp_A) for single-target move should be unchanged."""
        from turnone.data.dataset import _canonicalize_action_slot
        team = [{"moves": ["Thunderbolt", "UNK", "UNK", "UNK"],
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] + [{"moves": ["UNK"]*4,
                 "species": "A", "item": "X", "ability": "Y",
                 "tera_type": "Z"}] * 5
        result = _canonicalize_action_slot(0, team, 0)
        assert result == 0
