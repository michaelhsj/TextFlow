# Terraform setup for TextFlow

This Terraform setup deploys a Google Compute Engine instance.

## Prerequisites

- [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli) installed
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured

## Usage

1. Initialize Terraform:
   ```
   terraform init
   ```

2. Create a `terraform.tfvars` file with your project ID:
   ```
   project_id = "your-gcp-project-id"
   ```

   Optionally override `region`, `zone`, or `react_port` if you need values other than the defaults.

   The compute instance automatically pulls the image at
   `REGION-docker.pkg.dev/PROJECT/textflow-react/dev:tag`. Push a container
   image to that path after the repository is created and the VM will pick it
   up on the next pull attempt.

3. Apply the Terraform configuration:
   ```
   terraform apply
   ```

4. The IP address of the instance will be output after the apply is complete.

5. To destroy the infrastructure:
   ```
   terraform destroy
   ```
