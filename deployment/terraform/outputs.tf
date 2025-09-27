output "ip" {
  value = google_compute_instance.default.network_interface[0].access_config[0].nat_ip
}

output "workload_identity_provider_name" {
  value = google_iam_workload_identity_pool_provider.github_actions_provider.name
}

output "service_account_email" {
  value = google_service_account.github_actions_sa.email
}
