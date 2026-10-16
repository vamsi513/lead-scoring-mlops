"""Request and response models for the scoring API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    EVENT_TYPES,
    NO_EVENT_RECENCY,
    NUMERIC_FEATURES,
)

#: Counts cannot be negative, and nothing in a 60 day window exceeds a few
#: thousand events. The ceiling is a sanity bound, not a business rule.
_MAX_COUNT = 100_000.0

Count = Annotated[float, Field(ge=0.0, le=_MAX_COUNT)]


def _count_field(description: str) -> Count:
    return Field(ge=0.0, le=_MAX_COUNT, description=description)  # type: ignore[return-value]


class LeadFeatures(BaseModel):
    """One lead's features, as produced by the Spark job.

    Field names and meanings match ``lead_scoring.config.FEATURE_NAMES``
    exactly. The test in ``tests/test_serving.py`` fails if the two drift
    apart, so this cannot silently fall out of step with the model.
    """

    model_config = {"extra": "forbid"}

    n_email_open_7d: Count = _count_field("email opens in the 7 days before the cutoff")
    n_call_7d: Count = _count_field("calls in the 7 days before the cutoff")
    n_demo_request_7d: Count = _count_field("demo requests in the 7 days before the cutoff")
    n_page_visit_7d: Count = _count_field("page visits in the 7 days before the cutoff")
    n_total_7d: Count = _count_field("all events in the 7 days before the cutoff")

    n_email_open_30d: Count = _count_field("email opens in the 30 days before the cutoff")
    n_call_30d: Count = _count_field("calls in the 30 days before the cutoff")
    n_demo_request_30d: Count = _count_field("demo requests in the 30 days before the cutoff")
    n_page_visit_30d: Count = _count_field("page visits in the 30 days before the cutoff")
    n_total_30d: Count = _count_field("all events in the 30 days before the cutoff")

    n_total_prev7d: Count = _count_field("all events in days 14 to 7 before the cutoff")
    active_days_30d: Annotated[float, Field(ge=0.0, le=30.0)] = Field(
        description="distinct days with any event in the last 30"
    )
    distinct_types_30d: Annotated[float, Field(ge=0.0, le=float(len(EVENT_TYPES)))] = Field(
        description="distinct event types seen in the last 30 days"
    )
    recency_days: Annotated[float, Field(ge=0.0)] = Field(
        description=f"days since the last event; {NO_EVENT_RECENCY} when there is none"
    )
    trend_ratio: Annotated[float, Field(gt=0.0)] = Field(
        description="(events in last 7d + 1) / (events in the 7d before that + 1)"
    )
    trend_excess: float = Field(
        description="events in last 7d minus 7/30 of the 30 day count"
    )
    tenure_days: Annotated[float, Field(ge=0.0)] = Field(
        description="days between the lead being created and the cutoff"
    )

    industry: str
    company_size: str
    source: str
    region: str


class PredictRequest(BaseModel):
    """A batch of leads to score."""

    model_config = {"extra": "forbid"}

    leads: Annotated[list[LeadFeatures], Field(min_length=1, max_length=1000)]


class Prediction(BaseModel):
    index: int = Field(description="position of this lead in the request")
    probability: float = Field(
        ge=0.0, le=1.0, description="probability the lead converts within the label horizon"
    )


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    data_hash: str = Field(description="sha256 of the feature and label data used to fit")
    git_commit: str
    model_family: str
    train_cutoff: str
    test_cutoff: str
    data_is_synthetic: bool
    feature_names: list[str]


#: Sanity check at import time: the request schema and the model's feature
#: contract must name exactly the same columns.
_SCHEMA_FIELDS = set(LeadFeatures.model_fields)
_EXPECTED = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
if _SCHEMA_FIELDS != _EXPECTED:
    raise RuntimeError(
        "LeadFeatures is out of step with the feature contract: "
        f"missing {_EXPECTED - _SCHEMA_FIELDS}, unexpected {_SCHEMA_FIELDS - _EXPECTED}"
    )
