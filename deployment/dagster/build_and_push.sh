#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./build_and_push.sh --project PROJECT_ID --region REGION [--tag TAG]

Builds the Dagster webserver Docker image for linux/amd64 and pushes it to
Artifact Registry. The resulting image matches the path Terraform expects:
  REGION-docker.pkg.dev/PROJECT_ID/textflow-dagster/webserver:TAG

Environment variables can also be used instead of flags:
  PROJECT_ID, REGION, TAG (defaults to "latest" when omitted).
USAGE
}

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-}"
TAG="${TAG:-latest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$REGION" ]]; then
  echo "--project and --region are required (or set PROJECT_ID/REGION)." >&2
  usage
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/textflow-dagster/webserver:${TAG}"

echo "Building and pushing ${IMAGE} for linux/amd64..."

docker build \
  --platform linux/amd64 \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "$IMAGE" \
  "${REPO_ROOT}"

docker push "$IMAGE"

echo "Image pushed to ${IMAGE}"
