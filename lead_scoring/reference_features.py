"""Plain pandas reference implementation of the feature contract.

This exists only to check the PySpark job in ``pipelines/features.py``. It is
written independently, with pandas/NumPy idioms rather than a translation of
the Spark expressions, so that agreement between the two is evidence the
feature definitions are right and not evidence that one file imports the other.

It is deliberately the slow, obvious version: group, mask, count. Do not
optimise it, and do not refactor it to share code with the Spark job.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    COUNT_WINDOWS,
    EVENT_TYPES,
    NO_EVENT_RECENCY,
    NUMERIC_FEATURES,
)

_DAY_SECONDS = 86_400.0


def build_features_pandas(
    events: pd.DataFrame,
    leads: pd.DataFrame,
    cutoff: dt.datetime,
) -> pd.DataFrame:
    """Compute the feature table for ``cutoff`` using pandas only.

    ``events`` needs columns ``lead_id``, ``event_type``, ``event_ts``;
    ``leads`` needs ``lead_id``, ``created_at`` and the categorical columns.
    The returned frame is indexed 0..n-1 and sorted by ``lead_id``.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    before = events.loc[events["event_ts"] < cutoff_ts].copy()
    before["event_type"] = before["event_type"].astype(str)

    lead_ids = leads["lead_id"].to_numpy()
    out = pd.DataFrame({"lead_id": lead_ids})

    def counts_in(lower_days: int, upper_days: int) -> pd.DataFrame:
        """Rows in [cutoff - lower_days, cutoff - upper_days)."""
        lower = cutoff_ts - pd.Timedelta(days=lower_days)
        upper = cutoff_ts - pd.Timedelta(days=upper_days)
        mask = (before["event_ts"] >= lower) & (before["event_ts"] < upper)
        return before.loc[mask]

    for window in COUNT_WINDOWS:
        rows = counts_in(window, 0)
        for event_type in EVENT_TYPES:
            tally = rows.loc[rows["event_type"] == event_type].groupby("lead_id").size()
            out[f"n_{event_type}_{window}d"] = out["lead_id"].map(tally).fillna(0).astype("int32")
        total = rows.groupby("lead_id").size()
        out[f"n_total_{window}d"] = out["lead_id"].map(total).fillna(0).astype("int32")

    prev_rows = counts_in(14, 7)
    prev_total = prev_rows.groupby("lead_id").size()
    out["n_total_prev7d"] = out["lead_id"].map(prev_total).fillna(0).astype("int32")

    window30 = counts_in(30, 0).copy()
    window30["event_day"] = window30["event_ts"].dt.date
    active_days = window30.groupby("lead_id")["event_day"].nunique()
    out["active_days_30d"] = out["lead_id"].map(active_days).fillna(0).astype("int32")
    distinct_types = window30.groupby("lead_id")["event_type"].nunique()
    out["distinct_types_30d"] = out["lead_id"].map(distinct_types).fillna(0).astype("int32")

    last_event = before.groupby("lead_id")["event_ts"].max()
    last_mapped = out["lead_id"].map(last_event)
    recency = (cutoff_ts - last_mapped).dt.total_seconds() / _DAY_SECONDS
    out["recency_days"] = recency.fillna(NO_EVENT_RECENCY).astype("float64")

    out["trend_ratio"] = (out["n_total_7d"] + 1.0) / (out["n_total_prev7d"] + 1.0)
    out["trend_excess"] = out["n_total_7d"].astype("float64") - (7.0 / 30.0) * out[
        "n_total_30d"
    ].astype("float64")

    created = out["lead_id"].map(leads.set_index("lead_id")["created_at"])
    out["tenure_days"] = (cutoff_ts - created).dt.total_seconds() / _DAY_SECONDS

    for column in CATEGORICAL_FEATURES:
        out[column] = out["lead_id"].map(leads.set_index("lead_id")[column]).astype(str)

    ordered = ["lead_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    result: pd.DataFrame = out[ordered].sort_values("lead_id", ignore_index=True)
    assert not np.isnan(result["tenure_days"]).any(), "every lead needs a created_at"
    return result
