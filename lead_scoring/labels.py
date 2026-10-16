"""Label and eligibility rules for a snapshot.

Kept apart from the feature code on purpose. Features are built by the Spark
job from the event log and never see the conversions table; labels are built
here and joined on afterwards. That separation is what makes the leakage test
in ``tests/test_leakage.py`` meaningful.
"""

from __future__ import annotations

import pandas as pd

from lead_scoring.config import Snapshot


def build_labels(
    leads: pd.DataFrame,
    conversions: pd.DataFrame,
    snapshot: Snapshot,
) -> pd.DataFrame:
    """Return one row per *eligible* lead with a 0/1 ``converted`` label.

    A lead is eligible at a snapshot if it has not already converted before the
    cutoff: a lead who is already a customer is not a scoring candidate, and
    whether they converted is known at snapshot time, so excluding them is not
    leakage. Of the eligible leads, the label is 1 when the conversion falls in
    ``[cutoff, cutoff + horizon)`` and 0 otherwise, including when the lead
    converts later than the horizon.
    """
    converted_at = conversions.set_index("lead_id")["converted_at"]
    mapped = leads["lead_id"].map(converted_at)

    cutoff = pd.Timestamp(snapshot.cutoff)
    label_end = pd.Timestamp(snapshot.label_end)

    already_converted = mapped.notna() & (mapped < cutoff)
    in_window = mapped.notna() & (mapped >= cutoff) & (mapped < label_end)

    out = pd.DataFrame(
        {
            "lead_id": leads["lead_id"].to_numpy(),
            "converted": in_window.to_numpy().astype("int8"),
            "eligible": (~already_converted).to_numpy(),
        }
    )
    eligible: pd.DataFrame = out.loc[out["eligible"], ["lead_id", "converted"]].reset_index(
        drop=True
    )
    return eligible
