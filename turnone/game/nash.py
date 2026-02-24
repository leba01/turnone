"""Nash equilibrium solver for zero-sum games via linear programming.

Uses scipy.optimize.linprog to solve for minimax strategies.
For a zero-sum game with payoff matrix R (P1's reward):
  - P1 maximizes min_j (strategy @ R)_j
  - P2 minimizes max_i (R @ strategy)_i

These are dual LPs and have the same value (minimax theorem).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_nash_lp(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a zero-sum game for Nash equilibrium via LP.

    Args:
        R: (n1, n2) payoff matrix. R[i,j] = P1's reward when P1 plays i, P2 plays j.

    Returns:
        strategy_p1: (n1,) probability vector -- P1's Nash strategy.
        strategy_p2: (n2,) probability vector -- P2's Nash strategy.
        game_value: float -- value of the game (expected payoff under Nash play).

    The LP for P1 (maximizer):
        maximize v
        subject to: sum_i x_i * R[i,j] >= v  for all j
                    sum_i x_i = 1
                    x_i >= 0

    Rewritten for linprog (minimizer):
        minimize -v
        subject to: -R^T @ x + v*1 <= 0    (i.e. R^T @ x >= v)
                    sum(x) = 1
                    x >= 0

    Decision variables: [x_1, ..., x_n1, v]

    Similarly for P2 (by solving the transposed game).
    """
    # Solve for P1's strategy
    strategy_p1, value = _solve_maximin(R)

    # Solve for P2's strategy by solving the transposed game
    # P2 minimizes max_i (R @ y)_i, which is equivalent to
    # P2 maximizes min_j (-R^T @ y)_j, i.e., maximin on -R^T
    strategy_p2, neg_value = _solve_maximin(-R.T)

    # Sanity check: values should match (minimax theorem)
    if abs(value + neg_value) > 1e-3:
        import warnings
        warnings.warn(
            f"LP value mismatch: P1 maximin={value:.6f}, "
            f"P2 minimax={-neg_value:.6f}, gap={abs(value + neg_value):.6f}"
        )
    game_value = value

    return strategy_p1, strategy_p2, game_value


def _solve_maximin(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the maximin problem for a zero-sum game.

    maximize v
    subject to: R^T @ x >= v*1   (for each column j: sum_i x_i R_ij >= v)
                sum(x) = 1
                x >= 0

    Rewrite for linprog (minimization):
    Decision variables: z = [x_1, ..., x_n, v] (n+1 variables)

    minimize: c^T z = [0, ..., 0, -1] @ z = -v

    Inequality constraints (Ax <= b):
        For each column j: -sum_i x_i R_ij + v <= 0
        i.e., -R^T @ x + v*1 <= 0

        A_ub row j = [-R[0,j], -R[1,j], ..., -R[n1-1,j], 1]
        b_ub = 0 for all

    Equality constraints (Aeq @ z = beq):
        sum(x) = 1  ->  [1, 1, ..., 1, 0] @ z = 1

    Bounds: x_i >= 0, v unbounded (can be negative)
    """
    n1, n2 = R.shape
    n_vars = n1 + 1  # x_1..x_n1 + v

    # Objective: minimize -v
    c = np.zeros(n_vars)
    c[-1] = -1.0

    # Inequality: -R^T @ x + v <= 0
    # A_ub has shape (n2, n_vars)
    A_ub = np.zeros((n2, n_vars))
    A_ub[:, :n1] = -R.T  # -R^T
    A_ub[:, -1] = 1.0     # +v
    b_ub = np.zeros(n2)

    # Equality: sum(x) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n1] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x_i >= 0, v unbounded
    bounds = [(0, None)] * n1 + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        raise ValueError(f"LP solver failed: {result.message}")

    strategy = result.x[:n1]
    value = result.x[-1]

    # Clean up small negative values from numerical issues
    strategy = np.maximum(strategy, 0.0)
    strategy = strategy / strategy.sum()

    return strategy, value
