output "ip" {
  value = google_compute_instance.default.network_interface[0].access_config[0].nat_ip
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
  value       = var.enable_gpu_job_runner ? google_compute_instance.gpu_job_runner[0].network_interface[0].access_config[0].nat_ip : null
}
