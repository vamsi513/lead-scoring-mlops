"""Train and evaluate lead scoring models on the SYNTHETIC feature tables.

The split is by time, not at random: the model is fitted on the earlier
snapshot (features before 2025-01-31, label resolved in the following 15 days)
and evaluated on the later one (features before 2025-02-15, label resolved in
the 15 days after that). Nothing from the test snapshot's label window is
visible at fit time.

Two models are compared, a regularised logistic regression as the baseline and
histogram gradient boosting. Every run is recorded in a local MLflow file
store: parameters, metrics with bootstrap intervals, the hash of the exact
feature and label data used, and the git commit. The better model on test AUROC
is registered and given the "production" alias, which is what the serving app
loads.

Model scores here describe the synthetic generator in data/generate.py. They
are not evidence about real lead behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_scoring import cli
from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    CONVERSIONS_PATH,
    EXPERIMENT_NAME,
    FEATURE_NAMES,
    FEATURES_DIR,
    LEADS_PATH,
    MODEL_NAME,
    NUMERIC_FEATURES,
    PRODUCTION_ALIAS,
    REPO_ROOT,
    TEST_SNAPSHOT,
    TRAIN_SNAPSHOT,
    Snapshot,
)
from lead_scoring.labels import build_labels
from lead_scoring.metrics import (
    HIGHER_IS_BETTER,
    Interval,
    bootstrap_metrics,
    calibration_table,
)
from lead_scoring.provenance import git_commit, hash_paths

#: The metric the "best model" decision is made on.
SELECTION_METRIC: Final = "auroc"

#: Types MLflow's skops serialiser must be told to trust, per model family.
#:
#: skops refuses by default to write estimators whose reloading could be unsafe
#: with a hostile file: HistGradientBoosting stores raw node indices that
#: scikit-learn indexes without bounds checks. That warning is about loading
#: files from elsewhere. Here the estimator is fitted in this process, three
#: lines above the call, and written straight to the local store, so the type
#: is declared rather than the check disabled wholesale.
SKOPS_TRUSTED_TYPES: Final = {
    "hist_gradient_boosting": [
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
    ],
    "logistic_regression": [],
}

#: Local MLflow backend. Two local files, no tracking server and no network:
#: ``mlflow.db`` holds run and registry metadata, ``mlruns/`` holds artifacts.
#:
#: This is sqlite rather than the plain ``./mlruns`` file store because MLflow
#: 3.17.1 refuses to open a file store at all ("in maintenance mode"), and the
#: model registry, which the "production" alias requires, has never been
#: supported on it. sqlite is still a local file that no server process owns.
DEFAULT_DB_PATH: Final = REPO_ROOT / "mlflow.db"
DEFAULT_ARTIFACT_DIR: Final = REPO_ROOT / "mlruns"


def _ensure_experiment(artifact_dir: Path) -> None:
    """Select the experiment, creating it with an explicit artifact directory.

    ``set_experiment`` alone would put artifacts next to the sqlite file in a
    default location; naming the directory keeps the model files together under
    ``mlruns/`` where the Docker smoke test can mount them.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=str(artifact_dir))
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_snapshot(snapshot: Snapshot, features_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the feature table for a snapshot and attach its label."""
    features = pd.read_parquet(features_dir / f"snapshot={snapshot.cutoff_date}")
    leads = pd.read_parquet(LEADS_PATH, columns=["lead_id"])
    conversions = pd.read_parquet(CONVERSIONS_PATH)

    labels = build_labels(leads, conversions, snapshot)
    merged = features.merge(labels, on="lead_id", how="inner")
    merged = merged.sort_values("lead_id", ignore_index=True)

    x = merged[list(FEATURE_NAMES)].copy()
    # Count features are integers on disk, but the logged model signature is
    # what serving validates against, and JSON has no integer/float
    # distinction: a request sending 3.0 for a column typed as long fails
    # schema enforcement. Casting the numeric block to float64 here makes the
    # signature all-double and the API accepts either form.
    x[list(NUMERIC_FEATURES)] = x[list(NUMERIC_FEATURES)].astype("float64")
    y = merged["converted"].to_numpy().astype(np.int64)
    return x, y


def build_models(positive_rate: float) -> dict[str, Pipeline]:
    """Return the candidate models, each as a full preprocessing pipeline.

    Preprocessing lives inside the pipeline so the saved model takes raw
    feature columns, which is what the serving app sends it.
    """
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                # An unseen category at serving time becomes all-zeros rather
                # than an error, which is the right behaviour for a scoring API.
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )

    logistic = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    # Left unweighted on purpose: reweighting would improve
                    # ranking slightly and wreck calibration, and calibrated
                    # probabilities are what the serving API promises.
                    class_weight=None,
                    random_state=0,
                ),
            ),
        ]
    )

    boosting = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=400,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=0,
                ),
            ),
        ]
    )
    del positive_rate  # kept in the signature to document that no reweighting is applied
    return {"logistic_regression": logistic, "hist_gradient_boosting": boosting}


def _log_intervals(metrics: dict[str, Interval], prefix: str) -> None:
    """Log point estimate and both interval endpoints for each metric."""
    for name, interval in metrics.items():
        mlflow.log_metric(f"{prefix}_{name}", interval.value)
        mlflow.log_metric(f"{prefix}_{name}_ci_low", interval.low)
        mlflow.log_metric(f"{prefix}_{name}_ci_high", interval.high)


def train_one(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    data_hash: str,
    commit: str,
    n_boot: int,
) -> tuple[str, dict[str, Interval]]:
    """Fit one model, evaluate it on the later snapshot, log the run.

    Returns the MLflow run id and the test metrics.
    """
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "model_family": name,
                "git_commit": commit,
                "data_hash": data_hash,
                "data_is_synthetic": "true",
                "split": "time",
            }
        )
        mlflow.log_params(
            {
                "model": name,
                "train_cutoff": TRAIN_SNAPSHOT.cutoff_date,
                "test_cutoff": TEST_SNAPSHOT.cutoff_date,
                "label_horizon_days": TRAIN_SNAPSHOT.horizon_days,
                "n_train": len(x_train),
                "n_test": len(x_test),
                "train_positive_rate": round(float(y_train.mean()), 6),
                "test_positive_rate": round(float(y_test.mean()), 6),
                "n_features": len(FEATURE_NAMES),
                "bootstrap_resamples": n_boot,
                **{
                    f"model__{k}": v
                    for k, v in model.named_steps["model"].get_params().items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            }
        )

        model.fit(x_train, y_train)

        train_prob = model.predict_proba(x_train)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]

        train_metrics = bootstrap_metrics(y_train, train_prob, n_boot=n_boot)
        test_metrics = bootstrap_metrics(y_test, test_prob, n_boot=n_boot)
        _log_intervals(train_metrics, "train")
        _log_intervals(test_metrics, "test")

        table = calibration_table(y_test, test_prob)
        mlflow.log_text(
            json.dumps(
                [
                    {
                        "bin_low": round(low, 3),
                        "mean_predicted": round(pred, 5),
                        "observed_rate": round(obs, 5),
                        "count": count,
                    }
                    for low, pred, obs, count in table
                ],
                indent=2,
            ),
            "calibration_test.json",
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_train.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[name],
        )
        return run.info.run_id, test_metrics


def register_best(
    runs: dict[str, tuple[str, dict[str, Interval]]],
    data_hash: str,
    commit: str,
) -> tuple[str, int]:
    """Register the best run's model and move the production alias onto it."""
    best_name = max(
        runs,
        key=lambda name: runs[name][1][SELECTION_METRIC].value
        * (1 if SELECTION_METRIC in HIGHER_IS_BETTER else -1),
    )
    best_run_id = runs[best_name][0]

    best_metrics = runs[best_name][1]
    version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=MODEL_NAME,
        tags={
            "selected_on": SELECTION_METRIC,
            "model_family": best_name,
            # Read back by /model-info, so the served model can always say
            # which data it was fitted on.
            "data_hash": data_hash,
            "git_commit": commit,
            "data_is_synthetic": "true",
            "train_cutoff": TRAIN_SNAPSHOT.cutoff_date,
            "test_cutoff": TEST_SNAPSHOT.cutoff_date,
            "test_auroc": f"{best_metrics['auroc'].value:.4f}",
        },
    )
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version.version)
    return best_name, int(version.version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train lead scoring models (synthetic data).")
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="bootstrap resamples per metric"
    )
    args = parser.parse_args(argv)

    x_train, y_train = load_snapshot(TRAIN_SNAPSHOT, args.features)
    x_test, y_test = load_snapshot(TEST_SNAPSHOT, args.features)

    data_hash = hash_paths(
        [
            args.features / f"snapshot={TRAIN_SNAPSHOT.cutoff_date}",
            args.features / f"snapshot={TEST_SNAPSHOT.cutoff_date}",
            CONVERSIONS_PATH,
        ]
    )
    commit = git_commit(REPO_ROOT)

    tracking_uri = f"sqlite:///{args.db.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    _ensure_experiment(args.artifacts.resolve())

    print("SYNTHETIC data. Scores describe the generator, not real leads.")
    print(f"train snapshot {TRAIN_SNAPSHOT.cutoff_date}: {len(x_train):,} rows, "
          f"{y_train.mean():.3%} positive")
    print(f"test  snapshot {TEST_SNAPSHOT.cutoff_date}: {len(x_test):,} rows, "
          f"{y_test.mean():.3%} positive")
    print(f"data_hash {data_hash[:16]}  git {commit[:12]}")

    results: dict[str, tuple[str, dict[str, Interval]]] = {}
    for name, model in build_models(float(y_train.mean())).items():
        run_id, test_metrics = train_one(
            name,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            data_hash,
            commit,
            args.bootstrap,
        )
        results[name] = (run_id, test_metrics)
        print(f"\n{name}  (run {run_id[:8]})")
        for metric, interval in test_metrics.items():
            print(f"  test {metric:<20s} {interval}")

    best_name, version = register_best(results, data_hash, commit)
    print(f"\nregistered {MODEL_NAME} v{version} from {best_name}, alias '{PRODUCTION_ALIAS}'")
    return 0


if __name__ == "__main__":
    # Not SystemExit: see lead_scoring.cli for the Arrow shutdown deadlock.
    cli.run(main)
