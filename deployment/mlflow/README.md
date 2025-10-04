# MLflow Container

This directory holds the assets to build and deploy the MLflow tracking server that backs TextFlow experiments in Google Cloud. The container uses an embedded SQLite backend persisted on a Compute Engine disk and serves artifacts from a Google Cloud Storage bucket.

### Configuring MLflow clients for the remote server

Export the following variables in your development shell or add them to your virtual environment activation script so MLflow runs push to the shared server:

```bash
export MLFLOW_TRACKING_URI='http://<textflow-uri>/mlflow'
export MLFLOW_TRACKING_USERNAME='<mlflow-basic-auth-username>'
export MLFLOW_TRACKING_PASSWORD='<mlflow-basic-auth-password>'
```

Replace the placeholders with the proper credentials from shamsimuhaimen@gmail.com.

Persist the exports by appending the same lines to your shell profile (e.g. `~/.zshrc` or `~/.bashrc`). After saving, reload the profile with `source ~/.zshrc` (or open a new terminal) so the variables are available to `mlflow` in python.

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
   ./build_and_push.sh --project "$PROJECT_ID" --region "$REGION"
   ```

   You can also export `PROJECT_ID`, `REGION` environment variables instead of supplying flags.

3. (Optional) Verify the push through the console or:

   ```bash
   gcloud artifacts docker images list \
     northamerica-northeast1-docker.pkg.dev/$PROJECT_ID/textflow-mlflow
   ```

Resart the textflow-service instance to deploy the image after pushing.