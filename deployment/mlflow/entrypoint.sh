#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${MLFLOW_DB_PATH:-/app/db/mlflow.db}"
DEFAULT_BACKEND_URI="sqlite:///${DB_PATH}"
BACKEND_URI="${MLFLOW_BACKEND_STORE_URI:-$DEFAULT_BACKEND_URI}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/app/artifacts}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"
ARTIFACTS_DESTINATION="${MLFLOW_ARTIFACTS_DESTINATION:-file://${ARTIFACT_ROOT}}"
EXPERIMENT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-mlflow-artifacts:/}"

mkdir -p "$(dirname "${DB_PATH}")" "${ARTIFACT_ROOT}"

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${EXPERIMENT_ARTIFACT_ROOT}" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --serve-artifacts \
  --host "${HOST}" \
  --port "${PORT}"
