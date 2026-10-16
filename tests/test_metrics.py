"""Metric and bootstrap behaviour, checked against cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from lead_scoring.metrics import (
    bootstrap_metrics,
    calibration_table,
    expected_calibration_error,
    max_calibration_gap,
)


def test_auroc_is_one_for_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert bootstrap_metrics(y, p, n_boot=20)["auroc"].value == pytest.approx(1.0)


def test_auroc_is_half_for_constant_scores() -> None:
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert bootstrap_metrics(y, p, n_boot=20)["auroc"].value == pytest.approx(0.5)


def test_auroc_inverts_when_scores_are_reversed() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    p = rng.random(400)
    forward = bootstrap_metrics(y, p, n_boot=20)["auroc"].value
    reversed_ = bootstrap_metrics(y, 1.0 - p, n_boot=20)["auroc"].value
    assert forward + reversed_ == pytest.approx(1.0)


def test_brier_is_zero_for_exact_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    assert bootstrap_metrics(y, y.astype(float), n_boot=20)["brier"].value == pytest.approx(0.0)


def test_ece_is_zero_when_predictions_match_observed_rate() -> None:
    """Half the rows at p=0.25 with a true rate of 0.25, half at p=0.75 likewise."""
    y = np.array([1] * 25 + [0] * 75 + [1] * 75 + [0] * 25)
    p = np.array([0.25] * 100 + [0.75] * 100)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_equals_the_gap_for_a_uniformly_wrong_model() -> None:
    """Every row predicted 0.9 while the observed rate is 0.4: ECE is 0.5."""
    y = np.array([1] * 40 + [0] * 60)
    p = np.full(100, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.5)


def test_max_gap_ignores_bins_below_one_percent() -> None:
    """A single badly-placed row must not dominate the reported worst bin.

    999 rows sit at p=0.05, one row sits at p=0.95 and did not convert. The
    lone bin holds 0.1% of the rows, below the 1% floor, so it is excluded and
    the worst qualifying gap is the populated bin's 0.05.
    """
    y = np.zeros(1000, dtype=int)
    p = np.array([0.05] * 999 + [0.95])
    assert max_calibration_gap(y, p) == pytest.approx(0.05, abs=1e-9)


def test_max_gap_reports_a_bad_bin_that_is_large_enough() -> None:
    """The counterpart: once the badly-placed bin clears the 1% floor it is
    reported, so the floor is a noise filter and not a way to hide a problem."""
    y = np.zeros(1000, dtype=int)
    p = np.array([0.05] * 900 + [0.95] * 100)
    assert max_calibration_gap(y, p) == pytest.approx(0.95, abs=1e-9)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=1500)
    p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, 1500), 0.01, 0.99)
    for name, interval in bootstrap_metrics(y, p, n_boot=300).items():
        assert interval.low <= interval.value <= interval.high, name
        assert interval.low < interval.high, f"{name} interval has zero width"


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    first = bootstrap_metrics(y, p, n_boot=100, seed=42)
    second = bootstrap_metrics(y, p, n_boot=100, seed=42)
    assert first == second


def test_bootstrap_interval_narrows_as_the_sample_grows() -> None:
    """More data should buy a tighter interval, which is the whole point of
    reporting one."""
    rng = np.random.default_rng(11)

    def width(n: int) -> float:
        y = rng.integers(0, 2, size=n)
        p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.25, n), 0.01, 0.99)
        interval = bootstrap_metrics(y, p, n_boot=300)["auroc"]
        return interval.high - interval.low

    assert width(8000) < width(500)


def test_rejects_single_class_input() -> None:
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_metrics(np.zeros(10), np.full(10, 0.3))


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_metrics(np.array([0, 1, 0]), np.array([0.1, 0.2]))


def test_calibration_table_counts_add_up_and_bins_are_ordered() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=2000)
    p = rng.random(2000)
    rows = calibration_table(y, p)
    assert sum(count for *_, count in rows) == 2000
    lows = [low for low, *_ in rows]
    assert lows == sorted(lows)
