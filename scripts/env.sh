# Source this before running Spark locally:  source scripts/env.sh
# Points JAVA_HOME at the vendored JDK and makes Spark's Python workers use the
# venv interpreter rather than whatever `python3` happens to be on PATH.
_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -d "$_repo_root/.jdk/Contents/Home" ]; then
  export JAVA_HOME="$_repo_root/.jdk/Contents/Home"   # macOS bundle layout
elif [ -d "$_repo_root/.jdk/bin" ]; then
  export JAVA_HOME="$_repo_root/.jdk"                 # Linux layout
fi

if [ -n "${JAVA_HOME:-}" ]; then
  export PATH="$JAVA_HOME/bin:$PATH"
fi

if [ -x "$_repo_root/.venv/bin/python" ]; then
  export PYSPARK_PYTHON="$_repo_root/.venv/bin/python"
  export PYSPARK_DRIVER_PYTHON="$_repo_root/.venv/bin/python"
fi

export MLFLOW_TRACKING_URI="sqlite:///$_repo_root/mlflow.db"
export MLFLOW_REGISTRY_URI="sqlite:///$_repo_root/mlflow.db"
export MLFLOW_DISABLE_AGENT_HINT=1
unset _repo_root
