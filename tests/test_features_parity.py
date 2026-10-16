"""The Spark feature job must agree with the pandas reference, column by column.

This is the test that justifies trusting the Spark job at all: two independent
implementations of the same written contract, compared on real generated data
rather than on a hand-built toy frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lead_scoring.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from lead_scoring.reference_features import build_features_pandas
from pipelines.features import build_features
from tests.conftest import SampleData

pytestmark = pytest.mark.spark

#: Integer count columns must match exactly; float columns are compared with a
#: tolerance that is still far tighter than any difference that would matter.
_FLOAT_COLUMNS = ("recency_days", "trend_ratio", "trend_excess", "tenure_days")
_FLOAT_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def both_implementations(spark, sample_data: SampleData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Spark job and the pandas reference over the same sample."""
    events = spark.read.parquet(str(sample_data.root / "events"))
    leads = spark.read.parquet(str(sample_data.root / "leads.parquet"))

    spark_out = (
        build_features(events, leads, sample_data.cutoff)
        .toPandas()
        .sort_values("lead_id", ignore_index=True)
    )
    pandas_out = build_features_pandas(sample_data.events, sample_data.leads, sample_data.cutoff)
    return spark_out, pandas_out


def test_same_leads_in_both(both_implementations) -> None:
    spark_out, pandas_out = both_implementations
    assert len(spark_out) == len(pandas_out) == 400
    assert spark_out["lead_id"].tolist() == pandas_out["lead_id"].tolist()


@pytest.mark.parametrize("column", [c for c in NUMERIC_FEATURES if c not in _FLOAT_COLUMNS])
def test_integer_feature_matches_exactly(both_implementations, column: str) -> None:
    spark_out, pandas_out = both_implementations
    mismatches = spark_out.index[spark_out[column].to_numpy() != pandas_out[column].to_numpy()]
    assert mismatches.empty, (
        f"{column}: {len(mismatches)} rows differ, "
        f"first few spark={spark_out.loc[mismatches[:5], column].tolist()} "
        f"pandas={pandas_out.loc[mismatches[:5], column].tolist()}"
    )


@pytest.mark.parametrize("column", _FLOAT_COLUMNS)
def test_float_feature_matches_within_tolerance(both_implementations, column: str) -> None:
    spark_out, pandas_out = both_implementations
    diff = spark_out[column].to_numpy() - pandas_out[column].to_numpy()
    assert abs(diff).max() < _FLOAT_TOLERANCE, f"{column}: max abs diff {abs(diff).max()}"


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_categorical_feature_matches(both_implementations, column: str) -> None:
    spark_out, pandas_out = both_implementations
    assert spark_out[column].tolist() == pandas_out[column].tolist()


def test_sample_exercises_the_interesting_branches(both_implementations) -> None:
    """A parity test over data where every feature is zero would prove nothing.

    This asserts the sample actually contains the cases the feature code
    branches on, so the parity assertions above are doing real work.
    """
    _, pandas_out = both_implementations
    assert (pandas_out["n_total_7d"] == 0).any(), "no lead is quiet in the 7-day window"
    assert (pandas_out["n_total_7d"] > 0).any(), "no lead is active in the 7-day window"
    assert (pandas_out["n_demo_request_30d"] > 0).any(), "no demo requests in the sample"
    assert pandas_out["active_days_30d"].max() > 1, "no lead active on multiple days"
    assert pandas_out["distinct_types_30d"].max() >= 3, "event types too concentrated"
    assert pandas_out["recency_days"].min() < 1.0, "no lead active just before the cutoff"
