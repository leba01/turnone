"""Tests for the reward function."""

from __future__ import annotations

import numpy as np
import pytest

from turnone.rl.reward import compute_reward, compute_reward_from_dynamics


class TestComputeReward:
    """Tests for compute_reward."""

    def test_zero_sum(self):
        """Swapping P1/P2 perspectives negates reward."""
        # Create scenario: P1 deals damage to opponents, takes some
        N = 5
        hp_delta = np.random.RandomState(42).randn(N, 4).astype(np.float32) * 20
        ko_flags = np.zeros((N, 4), dtype=np.float32)
        field = np.zeros((N, 5), dtype=np.float32)

        r1 = compute_reward(hp_delta, ko_flags, field, field)

        # Swap perspectives: [our_a, our_b, opp_a, opp_b] -> [opp_a, opp_b, our_a, our_b]
        hp_swapped = hp_delta[:, [2, 3, 0, 1]]
        ko_swapped = ko_flags[:, [2, 3, 0, 1]]
        # Also swap tailwind columns (index 3 and 4)
        field_swapped = field.copy()
        field_swapped[:, 3], field_swapped[:, 4] = field[:, 4].copy(), field[:, 3].copy()

        r2 = compute_reward(hp_swapped, ko_swapped, field_swapped, field_swapped)
        np.testing.assert_allclose(r1, -r2, atol=1e-6)

    def test_damage_advantage(self):
        """Dealing more damage than taking yields positive reward."""
        N = 1
        # We deal 50 HP damage to each opponent, take 0
        hp_delta = np.array([[0, 0, 50, 50]], dtype=np.float32)
        ko_flags = np.zeros((N, 4), dtype=np.float32)
        field = np.zeros((N, 5), dtype=np.float32)

        r = compute_reward(hp_delta, ko_flags, field, field)
        assert r[0] > 0, "Dealing damage should yield positive reward"

    def test_ko_heavy_weight(self):
        """KO component dominates with default weights."""
        N = 1
        hp_delta = np.zeros((N, 4), dtype=np.float32)
        ko_flags = np.array([[0, 0, 1, 0]], dtype=np.float32)  # We KO one opponent
        field = np.zeros((N, 5), dtype=np.float32)

        r = compute_reward(hp_delta, ko_flags, field, field, w_ko=3.0)
        assert r[0] > 0
        assert r[0] == pytest.approx(3.0)  # 1 KO scored * w_ko

    def test_no_change_zero_reward(self):
        """No HP change, no KOs, no field change -> zero reward."""
        N = 3
        hp_delta = np.zeros((N, 4), dtype=np.float32)
        ko_flags = np.zeros((N, 4), dtype=np.float32)
        field = np.zeros((N, 5), dtype=np.float32)

        r = compute_reward(hp_delta, ko_flags, field, field)
        np.testing.assert_allclose(r, 0.0, atol=1e-7)

    def test_field_advantage(self):
        """Setting tailwind gives positive reward."""
        N = 1
        hp_delta = np.zeros((N, 4), dtype=np.float32)
        ko_flags = np.zeros((N, 4), dtype=np.float32)
        field_before = np.zeros((N, 5), dtype=np.float32)
        field_after = np.zeros((N, 5), dtype=np.float32)
        field_after[0, 3] = 1.0  # We set tailwind

        r = compute_reward(hp_delta, ko_flags, field_before, field_after, w_field=0.5)
        assert r[0] > 0

    def test_batch_shapes(self):
        """Verify output shapes for various batch sizes."""
        for N in [1, 5, 100]:
            hp = np.zeros((N, 4), dtype=np.float32)
            ko = np.zeros((N, 4), dtype=np.float32)
            field = np.zeros((N, 5), dtype=np.float32)
            r = compute_reward(hp, ko, field, field)
            assert r.shape == (N,)

    def test_hp_component_range(self):
        """HP component alone should be in [-1, +1] for any valid HP deltas.

        Each mon can lose 0-100 HP, two mons per side -> max sum = 200.
        Normalized by 200 -> each side in [0, 1], difference in [-1, +1].
        """
        N = 1
        ko = np.zeros((N, 4), dtype=np.float32)
        field = np.zeros((N, 5), dtype=np.float32)

        # Worst case for P1: both our mons take 100 damage, opponents take 0
        hp_worst = np.array([[100, 100, 0, 0]], dtype=np.float32)
        r_worst = compute_reward(hp_worst, ko, field, field, w_hp=1.0, w_ko=0.0, w_field=0.0)
        assert r_worst[0] == pytest.approx(-1.0), f"Max damage to us should give HP=-1.0, got {r_worst[0]}"

        # Best case for P1: we deal 100 to each opponent, take 0
        hp_best = np.array([[0, 0, 100, 100]], dtype=np.float32)
        r_best = compute_reward(hp_best, ko, field, field, w_hp=1.0, w_ko=0.0, w_field=0.0)
        assert r_best[0] == pytest.approx(1.0), f"Max damage to them should give HP=+1.0, got {r_best[0]}"

        # Random HP deltas: HP component should always be in [-1, +1]
        rng = np.random.RandomState(99)
        hp_rand = rng.uniform(0, 100, size=(1000, 4)).astype(np.float32)
        r_rand = compute_reward(hp_rand, np.zeros((1000, 4), dtype=np.float32),
                                np.zeros((1000, 5), dtype=np.float32),
                                np.zeros((1000, 5), dtype=np.float32),
                                w_hp=1.0, w_ko=0.0, w_field=0.0)
        assert r_rand.min() >= -1.0 - 1e-6, f"HP component below -1: {r_rand.min()}"
        assert r_rand.max() <= 1.0 + 1e-6, f"HP component above +1: {r_rand.max()}"


class TestRewardError:
    """Tests for compute_reward_error in dynamics_metrics."""

    def test_perfect_predictions(self):
        """Perfect dynamics predictions should give zero reward error."""
        from turnone.eval.dynamics_metrics import compute_reward_error

        N = 50
        rng = np.random.RandomState(42)
        hp = rng.uniform(0, 50, (N, 4)).astype(np.float32)
        ko = (rng.rand(N, 4) > 0.9).astype(np.float32)
        field_before = np.zeros((N, 5), dtype=np.float32)
        field_after = np.zeros((N, 5), dtype=np.float32)
        field_after[:, 3] = (rng.rand(N) > 0.5).astype(np.float32)

        # Perfect predictions: logits that give exact KO probs
        # For binary 0/1, use large positive/negative logits
        ko_logits = np.where(ko > 0.5, 30.0, -30.0).astype(np.float32)

        err = compute_reward_error(
            hp_pred=hp, hp_true=hp,
            ko_logits=ko_logits, ko_true=ko,
            field_pred=field_after, field_true=field_after,
            field_before=field_before,
        )

        assert err["reward_mae"] < 0.05
        assert err["reward_correlation"] > 0.99

    def test_error_keys(self):
        """Should return all expected keys."""
        from turnone.eval.dynamics_metrics import compute_reward_error

        N = 10
        z = np.zeros((N, 4), dtype=np.float32)
        f = np.zeros((N, 5), dtype=np.float32)
        err = compute_reward_error(z, z, z, z, f, f, f)
        assert set(err.keys()) == {"reward_mae", "reward_rmse", "reward_correlation", "reward_bias"}


class TestComputeRewardFromDynamics:
    """Tests for compute_reward_from_dynamics."""

    def test_consistent_with_ground_truth(self):
        """When dynamics outputs match ground truth, rewards should match."""
        N = 10
        rng = np.random.RandomState(123)
        hp_delta = rng.randn(N, 4).astype(np.float32) * 20
        ko_flags = np.zeros((N, 4), dtype=np.float32)
        field_before = np.zeros((N, 5), dtype=np.float32)
        field_after = rng.randn(N, 5).astype(np.float32) * 0.5

        # Ground truth reward
        r_gt = compute_reward(hp_delta, ko_flags, field_before, field_after)

        # "Perfect dynamics" -- same hp_pred, ko_logits at -30 (sigmoid -> ~0), same field
        ko_logits = np.full((N, 4), -30.0, dtype=np.float32)  # sigmoid(-30) ~ 0
        r_dyn = compute_reward_from_dynamics(hp_delta, ko_logits, field_after, field_before)

        np.testing.assert_allclose(r_gt, r_dyn, atol=0.01)

    def test_sigmoid_conversion(self):
        """KO logits are properly converted via sigmoid."""
        N = 1
        hp = np.zeros((N, 4), dtype=np.float32)
        ko_logits = np.array([[30, 30, 30, 30]], dtype=np.float32)  # sigmoid ~ 1
        field = np.zeros((N, 5), dtype=np.float32)

        r = compute_reward_from_dynamics(hp, ko_logits, field, field, w_ko=3.0)
        # All 4 KOs, but 2 ours (bad) and 2 opponents (good), net = 0
        assert abs(r[0]) < 0.1
