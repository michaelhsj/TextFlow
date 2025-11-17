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

4. The gateway now uses a reserved static IP so the address remains consistent across rebuilds. The address is printed after `terraform apply`. To see it without applying, run:
   ```
   terraform show
   ```

5. DNS for `textflowocr.com` is hosted on Cloudflare. Export a token with DNS Edit permissions so Terraform can keep the `textflow`, `@`, and `www` records in sync with the static IP:
   ```
   export TF_VAR_cloudflare_api_token="<cloudflare-token-with-dns-edit>"
   ```

## GPU job runner (configurable)

The default apply now creates a T4-equipped Compute Engine VM with Docker, NVIDIA drivers, and an authenticated helper script for Artifact Registry pulls. To skip provisioning this runner, set `TF_VAR_enable_gpu_job_runner=false` (or pass `-var enable_gpu_job_runner=false`). Images intended for this runner can be pushed to the automatically-created Artifact Registry repository `textflow-jobs`.

The runner is provisioned with a static external IP (`textflow-gpu-runner-ip`) so you can inspect or debug workloads over SSH. OS Login is enforced; grant yourself (and any other admins) the `roles/compute.osAdminLogin` IAM role in the project using `gcloud projects add-iam-policy-binding` or the Cloud Console. After applying, connect with:

```
gcloud compute ssh textflow-gpu-runner \
  --project "$TF_VAR_project_id" \
  --zone "${TF_VAR_gpu_zone:-northamerica-northeast1-c}"
```

Once the instance exists, dispatch a containerized job from your workstation:

```
python deployment/scripts/submit_gpu_job.py \
  --project "$TF_VAR_project_id" \
  --zone "${TF_VAR_zone:-northamerica-northeast1-a}" \
  --image northamerica-northeast1-docker.pkg.dev/$TF_VAR_project_id/textflow-jobs/train:latest \
  --env RUN_ID=test-run-001 -- python train.py --epochs 5
```

The helper will log in to Artifact Registry via the instance service account, run the container with `--gpus all`, and stream stdout/stderr through the SSH session. Repeat `--env`, `--volume`, or adjust `--gpus` as needed.
   
