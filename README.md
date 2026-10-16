# lead-scoring-mlops

A small lead scoring pipeline that runs end to end on one machine: generate
activity events, aggregate them into per-lead features with PySpark, train and
compare two models with confidence intervals, register the winner, serve it
behind FastAPI in a container, and check the feature distribution for drift.

Every number in this README was measured on the runs recorded below. Nothing
here is estimated or carried over from somewhere else.

## SYNTHETIC DATA

**There is no real data in this project.** There are no real leads, no real
companies, no real people and no real CRM export. Every row is produced by
`data/generate.py` from a seeded random process described in that file's
docstring: latent lead quality drives daily activity, and conversion is drawn
from a daily hazard that reads the previous seven days of activity.

This matters for reading the results. The model scores below say the pipeline
recovers signal that `data/generate.py` deliberately put there. They are not
evidence about real lead behaviour, and they do not transfer to a real CRM.
See [What the score means](#what-the-score-does-and-does-not-mean).

## What it does

```
data/generate.py      synthetic leads, 2M activity events, conversions
        |
pipelines/features.py PySpark: per-lead counts, recency and trend as of a cutoff
        |
train.py              time split, 2 models, bootstrap CIs, MLflow, registry alias
        |
serving/app.py        FastAPI, loads models:/lead-scoring@production
        |
monitoring/drift.py   population stability index per feature
```

### The cutoff, and why there is no label leakage

The 60-day window is cut at two points. Features are built from events strictly
before a cutoff; the label is whether the lead converts in the 15 days after it.

```
day 0                        day 30         day 45         day 60
|----------------------------|--------------|--------------|
                              ^train cutoff  ^test cutoff
train: features from events < day 30, label = converted in [day 30, day 45)
test:  features from events < day 45, label = converted in [day 45, day 60)
```

The test snapshot is strictly later than the train snapshot, so the split is by
time rather than at random. Three things keep the label out of the features:

- the generator's conversion hazard for day *d* only reads activity from days
  `[d-7, d)`, so no label is a function of post-cutoff events;
- the Spark job filters on the cutoff before aggregating, and never reads the
  conversions table at all;
- `tests/test_leakage.py` bolts 10,000 fabricated post-cutoff events onto the
  sample, reruns the job, and requires byte-identical features.

A converted lead stops producing activity, so post-cutoff event volume is a
*consequence* of the label. A pipeline that leaked it would score suspiciously
well, which makes the leakage test worth having rather than ceremonial.

## Requirements

- Python 3.12 (3.12.15 here). Pinned in `pyproject.toml`.
- A Java 17 JDK for Spark, even in local mode. `scripts/setup_jdk.sh` downloads
  Temurin 17 into `./.jdk` so nothing outside this directory changes; delete
  the folder to undo. An existing `JAVA_HOME` is used if set.
- Docker, only for the serving image.

Every dependency is pinned to an exact version in `requirements.txt` and
`requirements-dev.txt`.

## Setup

```bash
git clone <your-fork> lead-scoring-mlops && cd lead-scoring-mlops
./scripts/setup_jdk.sh                  # Temurin 17 into ./.jdk, ~180 MB
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip==25.3
.venv/bin/pip install -r requirements-dev.txt
source scripts/env.sh                   # JAVA_HOME, PYSPARK_PYTHON, MLFLOW_TRACKING_URI
```

`scripts/env.sh` pins `PYSPARK_PYTHON` to the venv interpreter. Without it the
JVM launches workers with whatever `python3` is first on `PATH`, and Spark fails
part-way into a job with `PYTHON_VERSION_MISMATCH`.

## Run each step

```bash
source scripts/env.sh

# 1. Generate the synthetic dataset (50k leads, ~2M events, 60 days)
.venv/bin/python -m data.generate

# 2. Build features for both snapshots with Spark
.venv/bin/python -m pipelines.features

# 3. Train, evaluate, log to MLflow, register the best model
.venv/bin/python train.py

# 4. Population stability index between the two snapshots
.venv/bin/python -m monitoring.drift

# 5. Serve locally
.venv/bin/uvicorn serving.app:app --port 8000
curl -s localhost:8000/health
curl -s localhost:8000/model-info

# 6. Or build the image and smoke test the container
./scripts/smoke_container.sh
```

Each step takes `--help`. The generator takes `--seed`, `--leads`, `--events`
and `--days`; `train.py` takes `--bootstrap`; `monitoring/drift.py` takes
`--reference`, `--current` and `--json`.

### Browsing the MLflow runs

The backend is a local sqlite file, `mlflow.db`, with artifacts under `mlruns/`.
No server is involved in writing it. To read it in the UI:

```bash
.venv/bin/mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db"
```

This is sqlite rather than a plain `./mlruns` file store for two reasons, both
of which were hit rather than anticipated: MLflow 3.17.1 refuses to open a file
store at all ("in maintenance mode"), and the model registry that the
`production` alias requires has never been supported on one. It is still a
local file that no server process owns.

### Run the API in Docker

```bash
docker build -t lead-scoring-serving:test .
docker run -d --name lead-scoring -p 8000:8000 \
  -v "$PWD/mlruns:$PWD/mlruns:ro" \
  -v "$PWD/mlflow.db:$PWD/mlflow.db:ro" \
  -e "MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db" \
  lead-scoring-serving:test
```

The store is mounted at the **same absolute path it has on the host** because
MLflow records artifact locations as absolute paths; a different mount point
inside the container would not resolve. It is mounted read-only, and the app
copies the model into its own filesystem at startup. The model is not baked
into the image, so promoting a new version needs no rebuild.

## Results

From the run below. Reproduce with the commands in [Run each step](#run-each-step).

### Environment

| | |
|---|---|
| Host | macOS 27.0.1, arm64, 10 cores, 16 GB RAM |
| Python | 3.12.15 |
| Java | Temurin 17.0.20.1 (vendored in `./.jdk`) |
| Spark | 4.2.0, `local[*]` |
| Docker | 29.7.2 |
| Commit | `bf3f1c0e8051bbe8592f279cd8d157537c3e2068` |

### Data and pipeline

| Stage | Measured |
|---|---|
| Leads | 50,000 |
| Activity events | 2,004,913 over 60 days (2025-01-01 to 2025-03-02) |
| Converting leads | 11,689 (23.38% over the whole window) |
| Generation | 1.88 s wall |
| On disk | events 20 MB (60 date partitions), leads 416 KB, conversions 80 KB |
| Spark features | 8.09 s wall total, including JVM start |
| — train snapshot | 50,000 rows in 3.0 s |
| — test snapshot | 50,000 rows in 1.4 s |
| Feature table | 2.4 MB, 21 features per lead |
| Training | 42.63 s wall, both models, 1000 bootstrap resamples each |
| Train rows | 44,021 eligible leads, 6.613% positive |
| Test rows | 41,110 eligible leads, 6.809% positive |
| Data hash | `9e82572d51db4c003ec18393bee8f4b828bebb652e69617f546e2da9ae46c420` |

Eligible means not already converted at the cutoff. The 23.38% whole-window
conversion rate and the ~6.7% per-snapshot rate are not in conflict: the label
only counts conversions inside a 15-day horizon.

### Model metrics

Test snapshot, 1000 bootstrap resamples, 95% percentile intervals. Lower is
better for Brier, ECE and max gap.

| Model | AUROC | AUPRC | Brier | ECE | Max calibration gap |
|---|---|---|---|---|---|
| **logistic regression** (registered) | **0.7037** [0.6919, 0.7149] | **0.2824** [0.2657, 0.3000] | **0.0563** [0.0544, 0.0582] | 0.0072 [0.0051, 0.0097] | 0.0414 [0.0185, 0.0681] |
| hist gradient boosting | 0.6978 [0.6860, 0.7087] | 0.2707 [0.2539, 0.2898] | 0.0567 [0.0548, 0.0586] | **0.0071** [0.0049, 0.0095] | **0.0234** [0.0150, 0.0557] |

Baseline for AUPRC is the positive rate, 0.0681. A random ranker would score
AUROC 0.5 and AUPRC 0.068.

**The baseline wins, and the intervals overlap.** The AUROC gap is 0.0059 with
intervals that overlap almost entirely, so this is not a real difference in
ranking quality. Train-set numbers explain why boosting does not pull ahead:

| Model | Train AUROC | Test AUROC |
|---|---|---|
| logistic regression | 0.6685 [0.6580, 0.6804] | 0.7037 [0.6919, 0.7149] |
| hist gradient boosting | 0.7740 [0.7644, 0.7832] | 0.6978 [0.6860, 0.7087] |

Boosting fits the training snapshot far better and does not carry it across,
which is overfitting even with early stopping and `min_samples_leaf=50`. The
generator's hazard is a logistic function of a roughly linear combination of
these features, so there is little non-linear structure for trees to find, and
what they find is mostly noise. The logistic model scoring *higher* on test
than on train is not a bug either: the two snapshots are different data with
different difficulty, and the test snapshot's positive rate is higher.

### Calibration of the registered model

Equal-width bins on the test snapshot.

| Predicted | Mean predicted | Observed rate | Leads |
|---|---|---|---|
| 0.0–0.1 | 0.0509 | 0.0464 | 35,598 |
| 0.1–0.2 | 0.1344 | 0.1172 | 3,576 |
| 0.2–0.3 | 0.2421 | 0.2007 | 822 |
| 0.3–0.4 | 0.3448 | 0.3235 | 371 |
| 0.4–0.5 | 0.4431 | 0.4638 | 235 |
| 0.5–0.6 | 0.5480 | 0.4747 | 158 |
| 0.6–0.7 | 0.6508 | 0.6053 | 114 |
| 0.7–0.8 | 0.7488 | 0.6585 | 82 |
| 0.8–0.9 | 0.8509 | 0.8391 | 87 |
| 0.9–1.0 | 0.9513 | 0.9552 | 67 |

Predictions track observed rates across the range, mildly over-confident in the
middle. The bins above 0.4 hold a few hundred leads between them, so their
observed rates are noisy.

### Drift

`python -m monitoring.drift`, train snapshot as reference against the test
snapshot, both 50,000 rows:

| Feature | PSI | Verdict |
|---|---|---|
| tenure_days | 0.1969 | watch |
| n_total_30d | 0.0903 | stable |
| n_page_visit_30d | 0.0893 | stable |
| n_email_open_30d | 0.0823 | stable |
| active_days_30d | 0.0815 | stable |
| distinct_types_30d | 0.0754 | stable |
| recency_days | 0.0517 | stable |
| … 14 others | ≤ 0.0488 | stable |
| industry, company_size, source, region | 0.0000 | stable |

0 features above 0.25. `tenure_days` moving most is correct rather than
alarming: the test snapshot is 15 days later, so every lead is 15 days older.
The four categorical features are exactly 0.0000 because the lead population is
identical between snapshots; only the activity window moved.

### Does the drift detector work?

From `tests/test_drift.py`, 100 trials per row, 3,000 rows per sample, flagged
at PSI ≥ 0.25 on any feature. These are counts from a real run, printed by the
tests themselves:

| Scenario | Flagged | Mean worst PSI | Max worst PSI |
|---|---|---|---|
| Fresh sample, same distribution | **0 / 100** | 0.0084 | 0.0183 |
| Shifted 25% | **100 / 100** | 0.9781 | 1.1534 |
| Shifted 5% | 0 / 100 | 0.0502 | 0.0798 |

No false positives, no misses at a 25% shift, and clean separation: the worst
PSI ever seen on an unshifted sample (0.0183) is far below the mean on a
shifted one (0.9781). The 5% row is the honest caveat — a shift that small does
not trip the 0.25 threshold at this sample size, though it does lift PSI about
six-fold above noise. Whether 5% *should* raise an alert is a judgement about
the use case, not a property of the detector.

### Tests

86 tests, all passing, 13.89 s:

| File | Tests | What it checks |
|---|---|---|
| `test_features_parity.py` | 23 | Spark output equals an independent pandas implementation, column by column |
| `test_metrics.py` | 14 | metrics against cases with known answers; bootstrap behaviour |
| `test_generate.py` | 12 | generator reproducibility, hash stability, activity really predicts conversion |
| `test_serving.py` | 11 | endpoints, validation, unseen categories, score ordering |
| `test_drift.py` | 9 | the trial counts above, plus PSI edge cases |
| `test_labels.py` | 7 | label window boundaries and eligibility |
| `test_leakage.py` | 5 | features do not move when post-cutoff events are added |
| `test_docker_image.py` | 5 | the built image contains every module the app imports |

Markers let you skip the slow ones: `pytest -m "not spark and not docker"`.

### Serving image

| | |
|---|---|
| Image size | 1.3 GB (mostly MLflow, scikit-learn and SciPy; no PySpark, no JVM) |
| User | non-root, uid 10001 |
| Model load | through `models:/lead-scoring@production`, ~3 s after start |
| Smoke test | `/health` ok, `/predict` quiet lead 0.0181 vs active lead 0.4665 |

The image test is there because a serving image usually breaks on a missing
`COPY`, not a bad command. It walks the real import graph of `serving.app` and
imports every first-party module *inside the container*. Verified against a
deliberately broken image with `COPY lead_scoring/` removed: 4 of 5 modules
fail, so the test is not vacuous.

## What the score does and does not mean

**What AUROC 0.7037 on this data means:** the pipeline recovers a usable part of
the signal that `data/generate.py` put into the event stream, without leaking
the label. Given one converting and one non-converting lead, the model ranks
them correctly about 70% of the time *on data from this generator*.

**What it does not mean:**

- It says nothing about real lead conversion. The generator's hazard is a smooth
  logistic function of a handful of clean features with no missingness, no
  tracking bugs, no duplicate contacts, no salespeople working the list and
  changing the outcome, and no seasonality beyond a weekend dip. Real activity
  data has all of that.
- It is not an upper or lower bound for the same approach on real data. The
  number is a property of the generator's noise level, which was chosen by hand
  to be learnable but not separable.
- Comparing the two models here says little about which would win elsewhere.
  Boosting loses because the true relationship is close to linear *by
  construction*.
- The calibration numbers are similarly a property of this setup. A model can be
  well calibrated on synthetic data drawn from a stationary process and badly
  calibrated the moment the process moves.

The defensible claims are about the machinery: the cutoff holds, the two feature
implementations agree, the metrics come with intervals, the runs are traceable
to a data hash and a commit, the registered model is what gets served, and the
drift detector fires when the data moves and not when it does not.

## Limitations

- **Single machine Spark.** `local[*]` on 10 cores against 2M events, 20 MB of
  Parquet. Nothing has been run on a cluster and nothing here has been tried at
  a scale where shuffle, skew or partitioning strategy would matter. The job
  takes seconds; that is a statement about the data size, not about throughput.
- **No real traffic.** The API has been exercised by its tests and the smoke
  script. There is no load test, no latency measurement under concurrency, and
  no idea how it behaves with more than one request at a time.
- **No authentication.** No auth, no authorisation, no rate limiting, no TLS,
  no input quotas beyond a 1,000-lead batch cap. Not a service to expose.
- **CI has never run on GitHub.** `.github/workflows/ci.yml` was written against
  the local runs above and its steps match them command for command, but it has
  not executed on a runner. The first push is also its first run.
- **Features are rebuilt in full, not incrementally.** Each snapshot is a
  complete pass over the event log. There is no streaming path and no backfill
  story.
- **Only one data hash per run.** The hash covers the feature tables and the
  conversions file. It does not cover the raw event log, so regenerating raw
  data without rebuilding features would go unnoticed.
- **PSI thresholds are convention, not calibration.** 0.1 and 0.25 are the
  numbers everyone uses. The trial counts above measure what they do on this
  data; they are not tuned to a false-positive budget.
- **Drift monitoring watches features only.** Nothing here monitors label drift,
  prediction drift over time, or model performance decay, which is what would
  actually tell you to retrain.
- **The serving image is 1.3 GB.** Full MLflow is needed for the registry and
  the sklearn flavour; `mlflow-skinny` drops the sqlalchemy store this depends
  on. It has not been slimmed.
- **MLflow artifact paths are absolute**, which is why the container mounts the
  store at its host path. A tidier arrangement would need a tracking server or a
  real artifact store, both of which were out of scope for a local project.
- **A pyarrow deadlock is worked around, not fixed.** Batch entry points end the
  process explicitly instead of returning, because libarrow's global thread pool
  can hang forever in its destructor during C++ static finalisation. See
  `lead_scoring/cli.py` for the stack and the conditions. Reproducible on every
  pyarrow from 18.1.0 to 25.0.1 on this platform.
- **One seed.** All results come from seed 20250101. Nothing reruns the whole
  pipeline across seeds to put an interval on the metrics themselves; the
  bootstrap intervals quantify sampling noise in the test set, not variation in
  the data generation.

## Repository layout

```
data/generate.py              synthetic generator (CLI)
pipelines/features.py         PySpark feature job (CLI)
train.py                      training, evaluation, MLflow, registry (CLI)
serving/app.py                FastAPI app
serving/schemas.py            request and response models
monitoring/drift.py           PSI (CLI)
lead_scoring/config.py        paths, timeline, feature contract
lead_scoring/labels.py        label and eligibility rules
lead_scoring/metrics.py       metrics and bootstrap intervals
lead_scoring/reference_features.py  independent pandas implementation, for the parity test
lead_scoring/provenance.py    data hashing, git commit
lead_scoring/importgraph.py   import-graph walk, for the image test
lead_scoring/spark.py         local SparkSession factory
lead_scoring/cli.py           deterministic process exit
scripts/setup_jdk.sh          vendored Temurin 17
scripts/env.sh                environment for local runs
scripts/smoke_container.sh    build, start, call, tear down
```

Lint and types: `ruff check .`, `ruff format --check .`, `mypy` (strict, 19
modules, clean).
