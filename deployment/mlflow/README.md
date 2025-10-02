# MLflow Container

This directory holds the assets to run an MLflow tracking server inside Docker. The container ships with an embedded SQLite database and file-based artifact store so you can test locally before wiring everything up to Google Cloud resources.

## Image layout

- **Base image**: `python:3.11-slim` with MLflow 2.17.1 pre-installed.
- **Backend store**: SQLite database at `/app/db/mlflow.db` (overridden via `MLFLOW_BACKEND_STORE_URI` or `MLFLOW_DB_PATH`).
- **Artifact root**: Files served over HTTP and persisted under `/app/artifacts` (configurable with `MLFLOW_ARTIFACTS_DESTINATION`).
- **Entrypoint**: `entrypoint.sh` wires environment variables into the `mlflow server` command, enables artifact serving, and ensures directories exist before starting.

## Running locally

```bash
cd deployment/mlflow
mkdir -p local_data/db local_data/artifacts  # optional, docker will auto-create
docker compose up --build
```

Once the container is healthy, the tracking UI is available at http://localhost:65500 by default (substitute the host port if you changed `MLFLOW_HOST_PORT`).

Local runs write the database and artifacts into `deployment/mlflow/local_data/` (ignored by git) so you can stop/start containers without losing test metadata.

### Publishing the image for Terraform

The Compute Engine instance defined in `deployment/terraform` expects the image
`REGION-docker.pkg.dev/PROJECT/textflow-mlflow/server:latest`. Follow these
steps whenever you need to rebuild and publish it:

1. Authenticate Docker with the Artifact Registry host (run once per machine):

   ```bash
   gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev
   ```

2. Build and push the image using the helper script (defaults to tag `latest`):

   ```bash
   cd deployment/mlflow
   ./build_and_push.sh --project "$PROJECT_ID" --region "northamerica-northeast1"
   ```

   You can also export `PROJECT_ID`, `REGION`, or `TAG` environment variables
   instead of supplying flags.

3. (Optional) Verify the push:

   ```bash
   gcloud artifacts docker images list \
     northamerica-northeast1-docker.pkg.dev/$PROJECT_ID/textflow-mlflow
   ```

The helper script issues `docker build --platform linux/amd64` followed by
`docker push`, ensuring the published image matches the Compute Engine
architecture.

### Environment overrides

The entrypoint understands the following environment variables:

- `MLFLOW_HOST` (default `0.0.0.0` inside the container)
- `MLFLOW_PORT` (default `5000` inside the container)
- `MLFLOW_HOST_PORT` (compose-only; default `65500` on the host)
- `MLFLOW_DB_PATH` (default `/app/db/mlflow.db`)
- `MLFLOW_BACKEND_STORE_URI` (default `sqlite:///${MLFLOW_DB_PATH}`)
- `MLFLOW_ARTIFACT_ROOT` (default `/app/artifacts` inside the container)
- `MLFLOW_ARTIFACTS_DESTINATION` (default `file://${MLFLOW_ARTIFACT_ROOT}`; controls where served artifacts land)
- `MLFLOW_DEFAULT_ARTIFACT_ROOT` (default `mlflow-artifacts:/`; keeps experiment artifact URIs compatible with the built-in artifact proxy)

#### Common tweaks

- Bind the UI on a different host port:

  ```bash
  MLFLOW_HOST_PORT=5000 docker compose up --build
  ```

- Keep the container listening on 5000 but use a custom experiment name from a notebook:

  ```bash
  export MLFLOW_TRACKING_URI=http://localhost:65500
  export MLFLOW_EXPERIMENT_NAME=textflow-doctr
  ```

- Store artifacts somewhere else (for example, a mounted directory):

  ```bash
  MLFLOW_ARTIFACT_ROOT=/mnt/artifacts \
  MLFLOW_ARTIFACTS_DESTINATION=file:///mnt/artifacts \
  docker compose up --build
  ```

## Deploying to Compute Engine later

The container is ready to drop into Terraform-managed infrastructure. When promoting to GCE:

1. Push the built image to Artifact Registry or another container registry.
2. Point `MLFLOW_BACKEND_STORE_URI` to 
3. Point `MLFLOW_ARTIFACTS_DESTINATION` (and optionally `MLFLOW_DEFAULT_ARTIFACT_ROOT`) to a durable bucket or object store and make sure the instance/service account can access it.
4. Provide the same environment variables in  Terraform resources (instance template, managed instance group, etc.).

Until that wiring exists, the embedded SQLite database plus local artifact volume is perfect for experimentation but should not be used in production.
