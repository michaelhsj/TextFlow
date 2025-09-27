terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Derive the dev image reference the VM should pull from Artifact Registry
locals {
  react_image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.textflow-react.repository_id}/dev:latest"
}

# Runs the React container on a Container-Optimized VM and exposes the desired port
resource "google_compute_instance" "default" {
  name                      = "textflow-instance"
  machine_type              = "e2-standard-2"
  allow_stopping_for_update = true

  tags = ["textflow-react"]

  boot_disk {
    initialize_params {
      image = "projects/cos-cloud/global/images/family/cos-stable"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = google_service_account.textflow_instance_sa.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata = {
    "gce-container-declaration" = <<-EOT
      spec:
        containers:
          - name: textflow-ui
            image: ${local.react_image}
            ports:
              - name: http
                hostPort: ${var.react_port}
                containerPort: ${var.react_port}
      restartPolicy: Always
    EOT
    "google-logging-enabled"    = "true"
    "google-monitoring-enabled" = "true"
  }

  depends_on = [google_artifact_registry_repository.textflow-react]
}

# Instance service account used to pull images and publish logs/metrics
resource "google_service_account" "textflow_instance_sa" {
  account_id   = "textflow-instance-sa"
  display_name = "TextFlow Compute Engine"
}

# Grant the instance service account read access to Artifact Registry
resource "google_project_iam_member" "textflow_instance_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.textflow_instance_sa.email}"
}

# Allow the instance to write logs to Cloud Logging
resource "google_project_iam_member" "textflow_instance_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.textflow_instance_sa.email}"
}

# Permit the instance to publish Cloud Monitoring metrics
resource "google_project_iam_member" "textflow_instance_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.textflow_instance_sa.email}"
}

# Keep SSH access available for emergency maintenance
resource "google_compute_firewall" "ssh" {
  name    = "allow-ssh"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["textflow-react"]
}

# Open the React application's port to internet clients
resource "google_compute_firewall" "textflow_react" {
  name    = "allow-textflow-react"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = [tostring(var.react_port)]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["textflow-react"]
}

# Holds the React container images in Artifact Registry
resource "google_artifact_registry_repository" "textflow-react" {
  location      = var.region
  repository_id = "textflow-react"
  description   = "Docker repository for images published from the textflow_ui repo"
  format        = "DOCKER"
}

# Federated identity pool letting GitHub Actions authenticate without static keys
resource "google_iam_workload_identity_pool" "github_actions_pool_2" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions-pool-2"
  display_name              = "GitHub Actions Pool"
  description               = "Workload Identity Pool for GitHub Actions"
}

# Maps GitHub OIDC tokens into the workload identity pool for the UI repo
resource "google_iam_workload_identity_pool_provider" "github_actions_provider" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_actions_pool_2.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions-provider"
  display_name                       = "GitHub Actions Provider"
  description                        = "Workload Identity Pool Provider for GitHub Actions"
  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.actor"            = "assertion.actor"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
  }
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
  attribute_condition = "attribute.repository_owner == 'shamsimuhaimen'"
}

# Service account GitHub Actions impersonates when deploying
resource "google_service_account" "github_actions_sa" {
  project      = var.project_id
  account_id   = "github-actions-sa"
  display_name = "GitHub Actions Service Account"
}

# Allow the CI service account to push images into Artifact Registry
resource "google_project_iam_member" "artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_actions_sa.email}"
}

# Grant the workload identity pool permission to impersonate the CI service account
resource "google_service_account_iam_binding" "github_actions_sa_binding" {
  service_account_id = google_service_account.github_actions_sa.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions_pool_2.name}/attribute.repository/shamsimuhaimen/textflow_ui"
  ]
}
