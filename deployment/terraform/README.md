# Terraform setup for TextFlow

* This Terraform setup deploys a Google Compute Engine instance.
* The process for updating the React Docker image being run on the instance is:
   * Merge/Rebase a change on to `textflow_ui/main`.
   * A Github Action will build a Docker image and push it to GC Artifact Registry.
   * Compute Engine instance will pick it up on reboot.
* There is only one development environment, but we may separate to dev and prod if needed.
* Terraform state is stored in GCS bucket.

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
   
