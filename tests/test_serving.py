"""API behaviour, exercised against the real registered model.

These tests need a model in the local registry, so they are marked ``model``
and skipped with a clear message when training has not been run.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lead_scoring.config import FEATURE_NAMES, MODEL_NAME, PRODUCTION_ALIAS, REPO_ROOT

pytestmark = pytest.mark.model


def _registry_available() -> bool:
    return (REPO_ROOT / "mlflow.db").exists()


@pytest.fixture(scope="module")
def client() -> Any:
    if not _registry_available():
        pytest.skip("no mlflow.db; run train.py first")
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    from serving.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_lead() -> dict[str, Any]:
    """A realistic payload taken from the real feature table where possible."""
    import pandas as pd

    snapshot = next((REPO_ROOT / "data" / "features").glob("snapshot=*"), None)
    if snapshot is not None:
        row = pd.read_parquet(snapshot).iloc[0]
        return {
            name: row[name].item() if hasattr(row[name], "item") else row[name]
            for name in FEATURE_NAMES
        }
    return {
        "n_email_open_7d": 3.0,
        "n_call_7d": 1.0,
        "n_demo_request_7d": 0.0,
        "n_page_visit_7d": 4.0,
        "n_total_7d": 8.0,
        "n_email_open_30d": 12.0,
        "n_call_30d": 3.0,
        "n_demo_request_30d": 1.0,
        "n_page_visit_30d": 15.0,
        "n_total_30d": 31.0,
        "n_total_prev7d": 7.0,
        "active_days_30d": 14.0,
        "distinct_types_30d": 4.0,
        "recency_days": 0.5,
        "trend_ratio": 1.125,
        "trend_excess": 0.766,
        "tenure_days": 120.0,
        "industry": "saas",
        "company_size": "51-200",
        "source": "referral",
        "region": "na",
    }


def test_health_reports_the_model_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_reports_version_and_data_hash(client) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == MODEL_NAME
    assert body["alias"] == PRODUCTION_ALIAS
    assert body["model_version"].isdigit()
    # A real sha256, not the "unknown" fallback.
    assert len(body["data_hash"]) == 64
    assert body["data_is_synthetic"] is True
    assert body["feature_names"] == list(FEATURE_NAMES)


def test_predict_returns_a_probability_per_lead(client, sample_lead) -> None:
    response = client.post("/predict", json={"leads": [sample_lead, sample_lead]})
    assert response.status_code == 200
    body = response.json()
    assert [p["index"] for p in body["predictions"]] == [0, 1]
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["probability"] <= 1.0
    # Identical inputs must score identically.
    assert body["predictions"][0]["probability"] == body["predictions"][1]["probability"]


def test_predict_is_monotonic_in_demo_requests(client, sample_lead) -> None:
    """More demo requests should not lower the score.

    Demo requests are the strongest intent signal in the generator, so a model
    that scored them downward would be a real defect rather than a quirk.
    """
    quiet = {**sample_lead, "n_demo_request_7d": 0.0, "n_demo_request_30d": 0.0}
    keen = {**sample_lead, "n_demo_request_7d": 5.0, "n_demo_request_30d": 9.0}
    response = client.post("/predict", json={"leads": [quiet, keen]})
    assert response.status_code == 200
    scores = [p["probability"] for p in response.json()["predictions"]]
    assert scores[1] > scores[0], scores


def test_predict_rejects_an_unknown_field(client, sample_lead) -> None:
    response = client.post("/predict", json={"leads": [{**sample_lead, "surprise": 1}]})
    assert response.status_code == 422


def test_predict_rejects_a_missing_feature(client, sample_lead) -> None:
    incomplete = {k: v for k, v in sample_lead.items() if k != "n_total_7d"}
    assert client.post("/predict", json={"leads": [incomplete]}).status_code == 422


def test_predict_rejects_a_negative_count(client, sample_lead) -> None:
    response = client.post("/predict", json={"leads": [{**sample_lead, "n_call_7d": -1.0}]})
    assert response.status_code == 422


def test_predict_rejects_an_empty_batch(client) -> None:
    assert client.post("/predict", json={"leads": []}).status_code == 422


def test_unseen_category_is_handled_rather_than_erroring(client, sample_lead) -> None:
    """OneHotEncoder is configured to ignore unknown levels, so a new industry
    must score rather than 500."""
    response = client.post("/predict", json={"leads": [{**sample_lead, "industry": "aerospace"}]})
    assert response.status_code == 200
    assert 0.0 <= response.json()["predictions"][0]["probability"] <= 1.0


def test_predict_accepts_integers_for_count_fields(client, sample_lead) -> None:
    """JSON has no int/float distinction; both forms must work."""
    as_ints = {
        k: (int(v) if isinstance(v, float) and k != "trend_ratio" and k != "trend_excess" else v)
        for k, v in sample_lead.items()
    }
    assert client.post("/predict", json={"leads": [as_ints]}).status_code == 200


def test_openapi_schema_is_served(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
