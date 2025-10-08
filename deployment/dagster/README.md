# Dagster Container

This directory contains the assets to build and publish the Dagster webserver image that will host TextFlow's data pipelines in Google Cloud. The container bundles the `pipeline` project from this repository alongside Dagster so it can serve asset definitions directly from source.

### Publishing the image for Terraform

The deployment defined under `deployment/terraform` expects the image
`REGION-docker.pkg.dev/PROJECT/textflow-dagster/webserver:latest`. Build and push it with the helper script whenever you make changes to the `pipeline` codebase:

1. Authenticate Docker with the Artifact Registry host (run once per machine):

   ```bash
   gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev
   ```

2. Build and push the image (defaults to tag `latest`):

   ```bash
   cd deployment/dagster
   ./build_and_push.sh --project "$PROJECT_ID" --region "$REGION"
   ```

   You can also export `PROJECT_ID`, `REGION`, and `TAG` environment variables instead of passing flags.

3. (Optional) Confirm the new image:

   ```bash
   gcloud artifacts docker images list \
     northamerica-northeast1-docker.pkg.dev/$PROJECT_ID/textflow-dagster
   ```

Restart the TextFlow services instance after pushing so the updated container is pulled.

### Running work on the GPU runner

Dagster orchestrates GPU-bound work by talking to the Docker engine exposed on the optional `textflow-gpu-runner` instance. The webserver container receives three environment variables from Terraform:

- `GPU_RUNNER_DOCKER_URL`: Docker API endpoint (defaults to `tcp://textflow-gpu-runner:2375`)
- `DAGSTER_JOB_IMAGE`: Artifact Registry image Dagster should launch for jobs (`textflow-jobs/pipeline:latest`)
- `DAGSTER_DATASETS_DIR`: Host path on the GPU runner where datasets persist (`/opt/textflow/datasets`)

Make sure the GPU runner is provisioned (`terraform apply -var enable_gpu_job_runner=true`) and that the job image has been published (see `deployment/jobs`). Dagster assets will fail fast with a helpful message if the GPU runner is offline.
