"""Label and eligibility rules, including the cases that would be leakage."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from lead_scoring.config import Snapshot
from lead_scoring.labels import build_labels

CUTOFF = dt.datetime(2025, 2, 1, tzinfo=dt.UTC)
SNAPSHOT = Snapshot("t", CUTOFF, horizon_days=15)


def _leads(*lead_ids: int) -> pd.DataFrame:
    return pd.DataFrame({"lead_id": list(lead_ids)})


def _conversions(pairs: dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_id": list(pairs),
            "converted_at": pd.to_datetime(list(pairs.values()), utc=True),
        }
    )


def test_conversion_inside_the_horizon_is_positive() -> None:
    out = build_labels(_leads(1), _conversions({1: "2025-02-10"}), SNAPSHOT)
    assert out.loc[0, "converted"] == 1


def test_lead_that_never_converts_is_negative() -> None:
    out = build_labels(_leads(1), _conversions({}), SNAPSHOT)
    assert out.loc[0, "converted"] == 0


def test_conversion_after_the_horizon_is_negative() -> None:
    """The label is "converted within 15 days", not "ever converted"."""
    out = build_labels(_leads(1), _conversions({1: "2025-03-20"}), SNAPSHOT)
    assert out.loc[0, "converted"] == 0


def test_lead_already_converted_before_the_cutoff_is_dropped() -> None:
    out = build_labels(_leads(1, 2), _conversions({1: "2025-01-05", 2: "2025-02-03"}), SNAPSHOT)
    assert out["lead_id"].tolist() == [2]


def test_cutoff_instant_counts_as_inside_the_window() -> None:
    """The window is half-open [cutoff, cutoff + horizon), so the cutoff
    instant itself is a positive and not an exclusion."""
    out = build_labels(_leads(1), _conversions({1: "2025-02-01T00:00:00"}), SNAPSHOT)
    assert out["lead_id"].tolist() == [1]
    assert out.loc[0, "converted"] == 1


def test_horizon_end_is_excluded() -> None:
    out = build_labels(_leads(1), _conversions({1: "2025-02-16T00:00:00"}), SNAPSHOT)
    assert out.loc[0, "converted"] == 0


def test_output_has_one_row_per_eligible_lead() -> None:
    leads = _leads(*range(1, 11))
    conversions = _conversions({1: "2025-01-02", 2: "2025-02-05", 3: "2025-04-01"})
    out = build_labels(leads, conversions, SNAPSHOT)
    assert len(out) == 9
    assert out["lead_id"].is_unique
    assert set(out.columns) == {"lead_id", "converted"}
