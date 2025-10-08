# Dagster Job Image

This directory builds the container image Dagster uses when it launches work on the GPU runner. The image bundles the `pipeline` package and supporting ML utilities so remote runs can execute tasks such as dataset downloads without requiring the full TextFlow repo on the instance.

### Publishing the image

Terraform expects the artifact at
`REGION-docker.pkg.dev/PROJECT/textflow-jobs/pipeline:latest`. Build and push it whenever you make changes to the pipeline or ML modules relied on by remote jobs:

```bash
cd deployment/jobs
./build_and_push.sh --project "$PROJECT_ID" --region "$REGION"
```

Set the `TAG` flag (or export `TAG`) if you want to publish a specific version. After pushing, restart the GPU runner or let the next job pull the tag automatically.
