"""Tests for BC evaluation metrics."""

import numpy as np
import pytest

from turnone.eval.metrics import compute_bc_metrics, compute_bc_metrics_stratified


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(n: int = 20) -> tuple:
    """Create deterministic synthetic test data.

    Returns (logits_a, logits_b, logits_tera,
             action_a, action_b, tera_label, mask_a, mask_b).
    """
    rng = np.random.RandomState(42)

    logits_a = rng.randn(n, 16).astype(np.float32)
    logits_b = rng.randn(n, 16).astype(np.float32)
    logits_tera = rng.randn(n, 3).astype(np.float32)

    # Make some slots invalid
    mask_a = np.ones((n, 16), dtype=bool)
    mask_b = np.ones((n, 16), dtype=bool)
    mask_a[:, 3] = False  # slot 3 invalid for all
    mask_b[:, 7] = False

    # Set invalid slots to -inf in logits
    logits_a[~mask_a] = -np.inf
    logits_b[~mask_b] = -np.inf

    # Labels: pick from valid slots
    action_a = np.zeros(n, dtype=np.int64)  # slot 0 is always valid
    action_b = np.zeros(n, dtype=np.int64)
    action_a[-2:] = -1  # last 2 examples: mon A fainted
    action_b[-1] = -1   # last example: mon B fainted

    tera_label = np.zeros(n, dtype=np.int64)
    tera_label[:5] = 1   # 5 examples use tera A
    tera_label[5:8] = 2  # 3 examples use tera B

    return logits_a, logits_b, logits_tera, action_a, action_b, tera_label, mask_a, mask_b


# ---------------------------------------------------------------------------
# TestTopKAccuracy
# ---------------------------------------------------------------------------

class TestTopKAccuracy:
    def test_perfect_predictions(self):
        """When argmax matches labels, top-1 = 1.0."""
        n = 10
        logits_a = np.full((n, 16), -10.0, dtype=np.float32)
        logits_b = np.full((n, 16), -10.0, dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)

        # Labels
        action_a = np.arange(n, dtype=np.int64) % 16
        action_b = np.arange(n, dtype=np.int64) % 16

        # Make the true label slot have a large logit
        for i in range(n):
            logits_a[i, action_a[i]] = 10.0
            logits_b[i, action_b[i]] = 10.0

        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)
        tera_label = np.zeros(n, dtype=np.int64)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert m["top1_a"] == pytest.approx(1.0)
        assert m["top1_b"] == pytest.approx(1.0)
        assert m["top1_avg"] == pytest.approx(1.0)

    def test_top3_geq_top1(self):
        """Top-3 accuracy is always >= top-1 accuracy."""
        data = _make_test_data(30)
        m = compute_bc_metrics(*data)

        assert m["top3_a"] >= m["top1_a"] - 1e-9
        assert m["top3_b"] >= m["top1_b"] - 1e-9
        assert m["top3_avg"] >= m["top1_avg"] - 1e-9

    def test_skips_neg1_labels(self):
        """Examples with action = -1 are excluded from accuracy."""
        n = 6
        logits_a = np.full((n, 16), -10.0, dtype=np.float32)
        logits_b = np.full((n, 16), -10.0, dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)

        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)

        # All labels -1 for mon A: n_valid_a = 0
        action_a = np.full(n, -1, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)  # all slot 0
        for i in range(n):
            logits_b[i, 0] = 10.0  # perfect for B

        tera_label = np.zeros(n, dtype=np.int64)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert m["n_valid_a"] == 0.0
        assert m["n_valid_b"] == float(n)
        assert m["top1_a"] == 0.0  # no valid examples, defaults to 0
        assert m["top1_b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestNLL
# ---------------------------------------------------------------------------

class TestNLL:
    def test_confident_correct_gives_low_nll(self):
        """Large logit on correct label -> NLL near 0."""
        n = 8
        logits_a = np.full((n, 16), -100.0, dtype=np.float32)
        logits_b = np.full((n, 16), -100.0, dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)

        action_a = np.full(n, 5, dtype=np.int64)  # all label=5
        action_b = np.full(n, 2, dtype=np.int64)

        # Put huge logit on the correct class
        for i in range(n):
            logits_a[i, 5] = 50.0
            logits_b[i, 2] = 50.0

        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)
        tera_label = np.zeros(n, dtype=np.int64)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert m["nll_a"] < 0.01
        assert m["nll_b"] < 0.01

    def test_uniform_gives_expected_nll(self):
        """Uniform logits over k valid slots -> NLL approx log(k)."""
        n = 10
        k = 4  # 4 valid slots per mon

        # Only slots 0..3 are valid, rest are -inf
        logits_a = np.full((n, 16), -np.inf, dtype=np.float32)
        logits_a[:, :k] = 0.0  # uniform over k slots

        logits_b = np.full((n, 16), -np.inf, dtype=np.float32)
        logits_b[:, :k] = 0.0

        logits_tera = np.zeros((n, 3), dtype=np.float32)

        action_a = np.zeros(n, dtype=np.int64)  # label=0 (one of the k valid)
        action_b = np.ones(n, dtype=np.int64)    # label=1

        mask_a = np.zeros((n, 16), dtype=bool)
        mask_a[:, :k] = True
        mask_b = np.zeros((n, 16), dtype=bool)
        mask_b[:, :k] = True
        tera_label = np.zeros(n, dtype=np.int64)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        expected_nll = np.log(k)
        assert m["nll_a"] == pytest.approx(expected_nll, abs=1e-6)
        assert m["nll_b"] == pytest.approx(expected_nll, abs=1e-6)


# ---------------------------------------------------------------------------
# TestMaskCompliance
# ---------------------------------------------------------------------------

class TestMaskCompliance:
    def test_properly_masked_gives_1(self):
        """When invalid slots have -inf logits, compliance = 1.0."""
        n = 10
        logits_a = np.full((n, 16), -np.inf, dtype=np.float32)
        logits_a[:, :4] = 1.0  # only first 4 slots are valid

        mask_a = np.zeros((n, 16), dtype=bool)
        mask_a[:, :4] = True

        # Need full args for compute_bc_metrics
        logits_b = np.zeros((n, 16), dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)
        action_a = np.zeros(n, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)
        tera_label = np.zeros(n, dtype=np.int64)
        mask_b = np.ones((n, 16), dtype=bool)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert m["mask_compliance_a"] == pytest.approx(1.0, abs=1e-9)

    def test_unmasked_leaks_prob(self):
        """If an invalid slot has finite logits, compliance < 1.0."""
        n = 10
        # All slots have finite logits (uniform)
        logits_a = np.zeros((n, 16), dtype=np.float32)
        # But only 8 of 16 slots are valid
        mask_a = np.zeros((n, 16), dtype=bool)
        mask_a[:, :8] = True

        logits_b = np.zeros((n, 16), dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)
        action_a = np.zeros(n, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)
        tera_label = np.zeros(n, dtype=np.int64)
        mask_b = np.ones((n, 16), dtype=bool)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        # Uniform over 16 slots, 8 valid => compliance = 8/16 = 0.5
        assert m["mask_compliance_a"] == pytest.approx(0.5, abs=1e-6)
        assert m["mask_compliance_a"] < 1.0


# ---------------------------------------------------------------------------
# TestTera
# ---------------------------------------------------------------------------

class TestTera:
    def test_tera_accuracy(self):
        """Perfect tera predictions -> acc = 1.0."""
        n = 12
        tera_label = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1], dtype=np.int64)

        # Build logits so argmax matches label
        logits_tera = np.full((n, 3), -10.0, dtype=np.float32)
        for i in range(n):
            logits_tera[i, tera_label[i]] = 10.0

        # Other args (don't matter for tera)
        logits_a = np.zeros((n, 16), dtype=np.float32)
        logits_b = np.zeros((n, 16), dtype=np.float32)
        action_a = np.zeros(n, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)
        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert m["tera_acc"] == pytest.approx(1.0)

    def test_tera_nll(self):
        """Tera NLL is finite and positive."""
        n = 10
        tera_label = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        # Moderate logits (not perfectly confident)
        logits_tera = np.array([
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [0.0, -1.0, -1.0],
        ], dtype=np.float32)

        logits_a = np.zeros((n, 16), dtype=np.float32)
        logits_b = np.zeros((n, 16), dtype=np.float32)
        action_a = np.zeros(n, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)
        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)

        m = compute_bc_metrics(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        assert np.isfinite(m["tera_nll"])
        assert m["tera_nll"] > 0.0


# ---------------------------------------------------------------------------
# TestStratified
# ---------------------------------------------------------------------------

class TestStratified:
    def test_both_acted_subset(self):
        """Verify both_acted filters correctly."""
        data = _make_test_data(20)
        (logits_a, logits_b, logits_tera,
         action_a, action_b, tera_label, mask_a, mask_b) = data

        strat = compute_bc_metrics_stratified(*data)

        # both_acted: both != -1
        both_mask = (action_a != -1) & (action_b != -1)
        expected_n = int(both_mask.sum())

        # n_valid_a in both_acted should equal the number of both-acted examples
        # (since by definition, action_a != -1 for all of them)
        assert strat["both_acted"]["n_valid_a"] == float(expected_n)
        assert strat["both_acted"]["n_valid_b"] == float(expected_n)

    def test_subsets_cover_all(self):
        """tera_used + no_tera counts = total."""
        data = _make_test_data(20)
        (logits_a, logits_b, logits_tera,
         action_a, action_b, tera_label, mask_a, mask_b) = data

        strat = compute_bc_metrics_stratified(*data)

        n = len(action_a)
        n_tera_used = int((tera_label != 0).sum())
        n_no_tera = int((tera_label == 0).sum())

        assert n_tera_used + n_no_tera == n

        # n_valid_a in overall should match overall compute
        overall = compute_bc_metrics(*data)
        assert strat["overall"]["n_valid_a"] == overall["n_valid_a"]
        assert strat["overall"]["n_valid_b"] == overall["n_valid_b"]

    def test_partial_subset(self):
        """Partial subset has exactly the right examples."""
        data = _make_test_data(20)
        (logits_a, logits_b, logits_tera,
         action_a, action_b, tera_label, mask_a, mask_b) = data

        strat = compute_bc_metrics_stratified(*data)

        # In _make_test_data: action_a[-2:] = -1, action_b[-1] = -1
        # Partial = exactly one is -1:
        #   index -2: action_a=-1, action_b=0 => partial (a fainted, b alive)
        #   index -1: action_a=-1, action_b=-1 => both fainted => NOT partial
        # So partial has 1 example with a=-1, b valid
        assert strat["partial"]["n_valid_a"] == 0.0   # all partial a's are -1
        assert strat["partial"]["n_valid_b"] == 1.0   # the one partial example has b valid

    def test_empty_subset_returns_zeros(self):
        """When a subset is empty, all values are 0.0."""
        # Create data where no one uses tera
        n = 5
        logits_a = np.zeros((n, 16), dtype=np.float32)
        logits_b = np.zeros((n, 16), dtype=np.float32)
        logits_tera = np.zeros((n, 3), dtype=np.float32)
        action_a = np.zeros(n, dtype=np.int64)
        action_b = np.zeros(n, dtype=np.int64)
        tera_label = np.zeros(n, dtype=np.int64)  # all no-tera
        mask_a = np.ones((n, 16), dtype=bool)
        mask_b = np.ones((n, 16), dtype=bool)

        strat = compute_bc_metrics_stratified(
            logits_a, logits_b, logits_tera,
            action_a, action_b, tera_label,
            mask_a, mask_b,
        )

        # tera_used subset is empty
        for v in strat["tera_used"].values():
            assert v == 0.0
