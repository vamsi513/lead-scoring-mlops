#!/usr/bin/env bash
# Build the serving image, start it, call /health, /model-info and /predict,
# then stop and remove the container. Used by CI and reproducible by hand.
#
# The MLflow store is mounted at the SAME absolute path it has on the host,
# because MLflow records artifact locations as absolute paths. It is mounted
# read-only; the app copies the model into its own filesystem at startup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lead-scoring-serving:test}"
CONTAINER="${CONTAINER:-lead-scoring-smoke}"
PORT="${PORT:-18000}"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ ! -f "$REPO_ROOT/mlflow.db" ]; then
  echo "no mlflow.db at $REPO_ROOT; run train.py first" >&2
  exit 1
fi

cleanup
docker build -t "$IMAGE" "$REPO_ROOT"

docker run -d --name "$CONTAINER" -p "$PORT:8000" \
  -v "$REPO_ROOT/mlruns:$REPO_ROOT/mlruns:ro" \
  -v "$REPO_ROOT/mlflow.db:$REPO_ROOT/mlflow.db:ro" \
  -e "MLFLOW_TRACKING_URI=sqlite:///$REPO_ROOT/mlflow.db" \
  "$IMAGE" >/dev/null

echo "waiting for the model to load"
for attempt in $(seq 1 60); do
  body="$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"
  case "$body" in
    *'"model_loaded":true'*) echo "ready after ${attempt}s"; break ;;
  esac
  if [ "$attempt" -eq 60 ]; then
    echo "model never loaded. /health said: ${body:-<no response>}" >&2
    docker logs "$CONTAINER" >&2
    exit 1
  fi
  sleep 1
done

echo "--- /health ---"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health"; echo

echo "--- /model-info ---"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/model-info" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in ("model_name","model_version","alias","model_family","data_hash","data_is_synthetic")}, indent=2))'

echo "--- /predict ---"
response="$(curl -fsS --max-time 15 -X POST "http://127.0.0.1:$PORT/predict" \
  -H 'Content-Type: application/json' \
  --data-binary @"$REPO_ROOT/scripts/sample_predict.json")"
echo "$response" | python3 -m json.tool

# A quiet lead must score below a very active one, otherwise the served model
# is wired up wrong even though every endpoint answered.
python3 - "$response" <<'PY'
import json, sys
scores = [p["probability"] for p in json.loads(sys.argv[1])["predictions"]]
assert len(scores) == 2, scores
assert all(0.0 <= s <= 1.0 for s in scores), scores
assert scores[0] < scores[1], f"quiet lead scored above the active one: {scores}"
print(f"ordering ok: quiet={scores[0]:.4f} < active={scores[1]:.4f}")
PY

echo "smoke test passed"
