"""FastAPI service that scores leads with the registered model.

The model is resolved through the MLflow registry alias
``models:/lead-scoring@production``; nothing here hardcodes a version or a
path. Promoting a new version is therefore a registry operation, and the
service picks it up on restart.

The scores are produced by a model fitted on SYNTHETIC data. There is no
authentication and no rate limiting: this is not a service to expose.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from lead_scoring.config import (
    FEATURE_NAMES,
    MODEL_NAME,
    NUMERIC_FEATURES,
    PRODUCTION_ALIAS,
)
from serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

logger = logging.getLogger("lead_scoring.serving")

#: Where the registry lives. Defaults to a sqlite file next to the app, which
#: is what the Docker smoke test mounts in.
DEFAULT_TRACKING_URI: Final = "sqlite:///mlflow.db"


@dataclass
class LoadedModel:
    """A model resolved through the registry alias, plus its provenance."""

    predictor: Any
    version: str
    data_hash: str
    git_commit: str
    model_family: str
    train_cutoff: str
    test_cutoff: str
    data_is_synthetic: bool


def load_production_model() -> LoadedModel:
    """Resolve the production alias and load the model behind it.

    Loaded through the ``sklearn`` flavor rather than ``pyfunc`` because the
    API returns probabilities: a pyfunc sklearn classifier's ``predict``
    returns hard 0/1 labels, which is not what a lead score is.
    """
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)

    # Materialise the model into a directory this process owns. Loading in
    # place would make MLflow write a registered_model_meta file next to the
    # artifacts, which fails outright when the store is mounted read-only and
    # would otherwise let the container mutate the shared store. Copying first
    # keeps the mount read-only and the load self-contained.
    staging = tempfile.mkdtemp(prefix="lead-scoring-model-")
    predictor = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}", dst_path=staging
    )

    tags = version.tags or {}
    return LoadedModel(
        predictor=predictor,
        version=str(version.version),
        data_hash=tags.get("data_hash", "unknown"),
        git_commit=tags.get("git_commit", "unknown"),
        model_family=tags.get("model_family", "unknown"),
        train_cutoff=tags.get("train_cutoff", "unknown"),
        test_cutoff=tags.get("test_cutoff", "unknown"),
        data_is_synthetic=tags.get("data_is_synthetic", "true") == "true",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup.

    A failure is recorded rather than raised, so that /health can report the
    service as degraded and say why. A container that exited on a missing
    model would give an operator far less to work with.
    """
    try:
        app.state.model = load_production_model()
        app.state.load_error = None
        logger.info(
            "loaded %s v%s (%s)",
            MODEL_NAME,
            app.state.model.version,
            app.state.model.model_family,
        )
    except Exception as exc:  # noqa: BLE001 - reported through /health
        app.state.model = None
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("could not load model: %s", app.state.load_error)
    yield


app = FastAPI(
    title="Lead scoring (synthetic data)",
    description=(
        "Scores leads with a model fitted on synthetic CRM activity. "
        "No authentication; not intended to be exposed."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _require_model(request: Request) -> LoadedModel:
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded: {getattr(request.app.state, 'load_error', 'unknown')}",
        )
    return model


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness plus whether the model actually loaded.

    Returns 200 either way: the process is up. ``model_loaded`` is what tells
    a caller whether /predict will work.
    """
    model = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Which model is serving, and which data it was fitted on."""
    model = _require_model(request)
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=model.version,
        alias=PRODUCTION_ALIAS,
        data_hash=model.data_hash,
        git_commit=model.git_commit,
        model_family=model.model_family,
        train_cutoff=model.train_cutoff,
        test_cutoff=model.test_cutoff,
        data_is_synthetic=model.data_is_synthetic,
        feature_names=list(FEATURE_NAMES),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score a batch of leads.

    The returned probability is the model's estimate that the lead converts
    within the label horizon after the cutoff its features describe.
    """
    model = _require_model(request)

    frame = pd.DataFrame([lead.model_dump() for lead in payload.leads])
    # Column order must match the fitted pipeline, and the numeric block must
    # be float64 to satisfy the logged signature.
    frame = frame[list(FEATURE_NAMES)]
    frame[list(NUMERIC_FEATURES)] = frame[list(NUMERIC_FEATURES)].astype("float64")

    probabilities = model.predictor.predict_proba(frame)[:, 1]

    return PredictResponse(
        model_name=MODEL_NAME,
        model_version=model.version,
        predictions=[
            Prediction(index=i, probability=float(p)) for i, p in enumerate(probabilities)
        ],
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Turn a bad feature frame into a 422 rather than a 500."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})
