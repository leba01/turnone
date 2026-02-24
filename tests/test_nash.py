"""Tests for Nash LP solver and exploitability computation."""

from __future__ import annotations

import numpy as np
import pytest

from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import (
    exploitability,
    exploitability_from_nash,
    bc_strategy_from_logits,
    compute_strategy_values,
)


class TestNashLP:
    """Tests for solve_nash_lp."""

    def test_rock_paper_scissors(self):
        """Classic RPS: Nash is uniform (1/3, 1/3, 1/3), value = 0."""
        R = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64)

        p1, p2, v = solve_nash_lp(R)

        assert p1.shape == (3,)
        assert p2.shape == (3,)
        np.testing.assert_allclose(p1, [1/3, 1/3, 1/3], atol=1e-6)
        np.testing.assert_allclose(p2, [1/3, 1/3, 1/3], atol=1e-6)
        assert abs(v) < 1e-6

    def test_dominant_strategy(self):
        """Game with a dominant strategy: one action always best."""
        R = np.array([
            [3, 3],
            [1, 1],
        ], dtype=np.float64)

        p1, p2, v = solve_nash_lp(R)

        # P1 should play action 0 (always better)
        assert p1[0] > 0.99
        assert abs(v - 3.0) < 1e-6

    def test_pure_nash(self):
        """Game with a pure Nash equilibrium (saddle point)."""
        R = np.array([
            [3, 1],
            [2, 4],
        ], dtype=np.float64)

        p1, p2, v = solve_nash_lp(R)

        # Verify strategies are valid probability vectors
        assert abs(p1.sum() - 1.0) < 1e-6
        assert abs(p2.sum() - 1.0) < 1e-6
        assert np.all(p1 >= -1e-6)
        assert np.all(p2 >= -1e-6)

    def test_valid_probability_vectors(self):
        """Nash strategies should be valid probability distributions."""
        R = np.random.RandomState(42).randn(5, 4)
        p1, p2, v = solve_nash_lp(R)

        assert abs(p1.sum() - 1.0) < 1e-6
        assert abs(p2.sum() - 1.0) < 1e-6
        assert np.all(p1 >= -1e-6)
        assert np.all(p2 >= -1e-6)

    def test_minimax_theorem(self):
        """Value from P1's LP should equal value from P2's LP."""
        R = np.random.RandomState(123).randn(4, 3)
        p1, p2, v = solve_nash_lp(R)

        # P1's guaranteed value: min_j (p1 @ R)_j
        v1 = (p1 @ R).min()
        # P2's guaranteed value: max_i (R @ p2)_i
        v2 = (R @ p2).max()

        assert abs(v1 - v2) < 1e-4
        assert abs(v - v1) < 1e-4

    def test_large_matrix(self):
        """Solver handles large-ish matrices (100x100)."""
        R = np.random.RandomState(456).randn(100, 100)
        p1, p2, v = solve_nash_lp(R)

        assert p1.shape == (100,)
        assert p2.shape == (100,)
        assert abs(p1.sum() - 1.0) < 1e-5
        assert abs(p2.sum() - 1.0) < 1e-5


class TestExploitability:
    """Tests for exploitability computation."""

    def test_nash_unexploitable(self):
        """Nash strategy has zero exploitability."""
        R = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64)

        p1, p2, v = solve_nash_lp(R)

        e1 = exploitability_from_nash(p1, R, v, player=1)
        e2 = exploitability_from_nash(p2, R, v, player=2)

        assert e1 < 1e-6
        assert e2 < 1e-6

    def test_uniform_exploitable(self):
        """Uniform strategy on a non-symmetric game should be exploitable."""
        R = np.array([
            [3, 1],
            [0, 2],
        ], dtype=np.float64)

        uniform = np.array([0.5, 0.5])
        _, _, v = solve_nash_lp(R)

        e = exploitability_from_nash(uniform, R, v, player=1)
        assert e >= 0  # exploitability is non-negative


class TestStrategyValues:
    """Tests for compute_strategy_values (safety-exploitation triangle)."""

    def test_ordering(self):
        """BC worst-case <= Nash value <= best-response-to-BC."""
        R = np.random.RandomState(42).randn(10, 8).astype(np.float64)
        p1, p2, v = solve_nash_lp(R)

        # Use a somewhat suboptimal strategy
        bc_p1 = np.ones(10) / 10.0
        bc_p2 = np.ones(8) / 8.0

        vals = compute_strategy_values(bc_p1, bc_p2, R, v)

        # bc_worst_case <= nash_value (by definition of exploitability)
        assert vals["bc_worst_case"] <= vals["nash_value"] + 1e-6
        # nash_value <= best_response_to_bc (best response is at least as good as Nash)
        assert vals["nash_value"] <= vals["best_response_to_bc"] + 1e-6

    def test_nash_as_bc(self):
        """When BC = Nash, worst case = Nash value and exploit = 0."""
        R = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64)

        p1, p2, v = solve_nash_lp(R)
        vals = compute_strategy_values(p1, p2, R, v)

        assert abs(vals["bc_worst_case"] - vals["nash_value"]) < 1e-4
        assert abs(vals["bc_vs_bc"] - vals["nash_value"]) < 1e-4

    def test_all_keys_present(self):
        """Should return all four triangle values."""
        R = np.eye(3)
        bc_p1 = np.array([1/3, 1/3, 1/3])
        bc_p2 = np.array([1/3, 1/3, 1/3])
        vals = compute_strategy_values(bc_p1, bc_p2, R, 0.5)
        assert set(vals.keys()) == {"bc_worst_case", "nash_value", "bc_vs_bc", "best_response_to_bc"}


class TestBCStrategy:
    """Tests for bc_strategy_from_logits."""

    def test_output_is_distribution(self):
        """Output should be a valid probability distribution."""
        logits_a = np.random.randn(16).astype(np.float32)
        logits_b = np.random.randn(16).astype(np.float32)
        logits_tera = np.random.randn(3).astype(np.float32)

        # Enumerate some valid actions
        valid_actions = [(sa, sb, tf) for sa in range(4) for sb in range(4) for tf in range(3)]

        probs = bc_strategy_from_logits(logits_a, logits_b, logits_tera, valid_actions)

        assert probs.shape == (len(valid_actions),)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_peaked_logits(self):
        """High logits for specific actions should concentrate probability."""
        logits_a = np.full(16, -30.0, dtype=np.float32)
        logits_a[0] = 10.0  # slot 0 dominant
        logits_b = np.full(16, -30.0, dtype=np.float32)
        logits_b[4] = 10.0  # slot 4 dominant
        logits_tera = np.array([-30.0, -30.0, 10.0], dtype=np.float32)  # tera 2 dominant

        valid_actions = [(sa, sb, tf) for sa in range(16) for sb in range(16) for tf in range(3)]

        probs = bc_strategy_from_logits(logits_a, logits_b, logits_tera, valid_actions)

        # Find the index of (0, 4, 2)
        target_idx = valid_actions.index((0, 4, 2))
        assert probs[target_idx] > 0.9  # should be very concentrated
