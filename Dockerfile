# Serving image for the lead scoring API.
#
# The model is NOT baked in. It is resolved at startup through the MLflow
# registry alias, from a store mounted at runtime, so promoting a new version
# needs no rebuild. See README "Run the API in Docker" for the mount, and the
# Limitations section for why the mount path is what it is.
#
# Base image pinned by digest so a rebuild gets the same bytes.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

# Fail fast and keep the image free of .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dependencies first, so edits to the application code reuse this layer.
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

# Application code. Both packages are needed: serving imports lead_scoring for
# the feature contract and the registry names. tests/test_docker_image.py
# checks this list against the app's real import graph, so a package added
# later cannot be silently left out of the image.
COPY lead_scoring/ ./lead_scoring/
COPY serving/ ./serving/

# Run as a non-root user that owns nothing it does not need to write.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Defaults to a registry mounted at /app/mlflow.db; override at run time.
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
