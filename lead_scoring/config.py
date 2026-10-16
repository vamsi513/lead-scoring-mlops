"""Paths, time windows and the feature contract.

The timeline below is the single source of truth for every stage. Changing a
cutoff here changes the generator, the Spark job, training and monitoring
together, which is the only way the no-leakage guarantee stays true.

    day 0                       day 30        day 45        day 60
    |---------------------------|-------------|-------------|
    <----- events generated over 60 days ---------------->
                                 ^TRAIN cutoff ^TEST cutoff
    TRAIN features: events < day 30, label: converted in [day 30, day 45)
    TEST  features: events < day 45, label: converted in [day 45, day 60)

The test snapshot is strictly later in time than the train snapshot, so the
train/test split is a time split and not a random one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent

RAW_DIR: Final = REPO_ROOT / "data" / "raw"
EVENTS_DIR: Final = RAW_DIR / "events"
LEADS_PATH: Final = RAW_DIR / "leads.parquet"
CONVERSIONS_PATH: Final = RAW_DIR / "conversions.parquet"
FEATURES_DIR: Final = REPO_ROOT / "data" / "features"

# --- timeline -----------------------------------------------------------------

#: First day of the synthetic observation window (UTC midnight).
DAY_ZERO: Final = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

#: Length of the generated event window.
OBSERVATION_DAYS: Final = 60

#: Days from a snapshot cutoff in which a conversion counts as a positive label.
LABEL_HORIZON_DAYS: Final = 15


@dataclass(frozen=True)
class Snapshot:
    """One point in time at which features are computed and a label is resolved."""

    name: str
    cutoff: dt.datetime
    horizon_days: int = LABEL_HORIZON_DAYS

    @property
    def label_end(self) -> dt.datetime:
        return self.cutoff + dt.timedelta(days=self.horizon_days)

    @property
    def cutoff_date(self) -> str:
        return self.cutoff.date().isoformat()


TRAIN_SNAPSHOT: Final = Snapshot("train", DAY_ZERO + dt.timedelta(days=30))
TEST_SNAPSHOT: Final = Snapshot("test", DAY_ZERO + dt.timedelta(days=45))
SNAPSHOTS: Final = (TRAIN_SNAPSHOT, TEST_SNAPSHOT)

# --- generator size -----------------------------------------------------------

N_LEADS: Final = 50_000
TARGET_EVENTS: Final = 2_000_000
DEFAULT_SEED: Final = 20250101

# --- feature contract ---------------------------------------------------------

#: Activity event types the CRM emits.
EVENT_TYPES: Final = ("email_open", "call", "demo_request", "page_visit")

#: Lookback windows, in days, for per-type and total counts.
COUNT_WINDOWS: Final = (7, 30)

#: Value used for ``recency_days`` when a lead has no event before the cutoff.
#: Larger than any possible real recency inside the observation window, so it
#: orders correctly as "least recent" for both trees and linear models.
NO_EVENT_RECENCY: Final = 999.0

CATEGORICAL_FEATURES: Final = ("industry", "company_size", "source", "region")


def numeric_feature_names() -> tuple[str, ...]:
    """Numeric feature column names, in a fixed order.

    The order is part of the contract: the Spark job, the pandas reference
    implementation, training and the serving payload all rely on it.
    """
    names: list[str] = []
    for window in COUNT_WINDOWS:
        names.extend(f"n_{event_type}_{window}d" for event_type in EVENT_TYPES)
        names.append(f"n_total_{window}d")
    names.extend(
        [
            "n_total_prev7d",
            "active_days_30d",
            "distinct_types_30d",
            "recency_days",
            "trend_ratio",
            "trend_excess",
            "tenure_days",
        ]
    )
    return tuple(names)


NUMERIC_FEATURES: Final = numeric_feature_names()
FEATURE_NAMES: Final = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_NAME: Final = "lead-scoring"
PRODUCTION_ALIAS: Final = "production"
EXPERIMENT_NAME: Final = "lead-scoring"
