# Terraform setup for TextFlow

* This Terraform setup deploys a Google Compute Engine instance.
* The process for updating the React Docker image being run on the instance is:
   * Merge/Rebase a change on to `textflow_ui/main`.
   * A Github Action will build a Docker image and push it to GC Artifact Registry.
   * Compute Engine instance will pick it up on reboot.
* There is only one development environment, but we may separate to dev and prod if needed.
* Terraform state is stored in GCS bucket.
* Provisions a GPU-backed "job runner" VM for ad-hoc containerized training/evaluation workloads (can be disabled).

## Prerequisites

- [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli) installed
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured

## Usage

1. Initialize Terraform:
   ```
   terraform init
   ```

2. Export the project ID as an environment variable before planning or applying:
   ```
   export TF_VAR_project_id="your-gcp-project-id"
   ```

3. Apply the Terraform configuration:
   ```
   terraform apply
   ```

4. The IP address of the instance will be output after the apply is complete. To see it without applying, run:
   ```
   terraform show
   ```

## GPU job runner (configurable)

The default apply now creates a T4-equipped Compute Engine VM with Docker, NVIDIA drivers, and an authenticated helper script for Artifact Registry pulls. To skip provisioning this runner, set `TF_VAR_enable_gpu_job_runner=false` (or pass `-var enable_gpu_job_runner=false`). Images intended for this runner can be pushed to the automatically-created Artifact Registry repository `textflow-jobs`.

Once the instance exists, dispatch a containerized job from your workstation:

```
python deployment/scripts/submit_gpu_job.py \
  --project "$TF_VAR_project_id" \
  --zone "${TF_VAR_zone:-northamerica-northeast1-a}" \
  --image northamerica-northeast1-docker.pkg.dev/$TF_VAR_project_id/textflow-jobs/train:latest \
  --env RUN_ID=test-run-001 -- python train.py --epochs 5
```

The helper will log in to Artifact Registry via the instance service account, run the container with `--gpus all`, and stream stdout/stderr through the SSH session. Repeat `--env`, `--volume`, or adjust `--gpus` as needed.
   
