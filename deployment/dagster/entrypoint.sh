#!/usr/bin/env bash
set -euo pipefail

DAGSTER_HOME="${DAGSTER_HOME:-/opt/dagster/dagster_home}"
DAGSTER_PROJECT_DIR="${DAGSTER_PROJECT_DIR:-/opt/dagster/app}"
DAGSTER_MODULE="${DAGSTER_MODULE:-pipeline.definitions}"
DAGSTER_PORT="${DAGSTER_PORT:-3000}"
DAGSTER_PATH_PREFIX="${DAGSTER_PATH_PREFIX:-}"
DAGSTER_CONFIG_SOURCE="${DAGSTER_PROJECT_DIR}/dagster.yaml"
DAGSTER_CONFIG_DEST="${DAGSTER_HOME}/dagster.yaml"

mkdir -p "${DAGSTER_HOME}"

# Copy the default dagster.yaml into the volume if it is missing so Dagster can start.
if [[ -f "${DAGSTER_CONFIG_SOURCE}" && ! -f "${DAGSTER_CONFIG_DEST}" ]]; then
  cp "${DAGSTER_CONFIG_SOURCE}" "${DAGSTER_CONFIG_DEST}"
fi

export PYTHONPATH="${DAGSTER_PROJECT_DIR}:${PYTHONPATH:-}"

cmd=(dagster-webserver \
  --host 0.0.0.0 \
  --port "${DAGSTER_PORT}" \
  --module-name "${DAGSTER_MODULE}")

if [[ -n "${DAGSTER_PATH_PREFIX}" ]]; then
  cmd+=(--path-prefix "${DAGSTER_PATH_PREFIX}")
fi

exec "${cmd[@]}"
