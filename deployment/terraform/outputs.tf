output "ip" {
  value = google_compute_address.gateway_static.address
}

output "workload_identity_provider_name" {
  value = google_iam_workload_identity_pool_provider.github_actions_provider.name
}

output "service_account_email" {
  value = google_service_account.github_actions_sa.email
}

output "mlflow_artifact_bucket" {
  description = "GCS bucket backing MLflow artifact persistence."
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "gpu_runner_ip" {
  description = "External IP for the optional GPU job runner instance."
  value       = var.enable_gpu_job_runner ? google_compute_address.gpu_job_runner_static[0].address : null
}
