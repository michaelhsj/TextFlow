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

terraform {
  backend "gcs" {}
}

# Resolve the latest Ubuntu LTS image for the main instance
data "google_compute_image" "ubuntu_2204_lts" {
  family  = "ubuntu-2204-lts"
  project = "ubuntu-os-cloud"
}

locals {
  # Docker Images
  react_image   = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.textflow-react.repository_id}/dev:latest"
  mlflow_image  = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.textflow_mlflow.repository_id}/server:latest"
  gateway_image = "nginx:1.25-alpine"

  # Storage
  perma_disk_mount_path     = "/mnt/disks/perma-disk"
  perma_db_host_path        = "${local.perma_disk_mount_path}/db"
  perma_artifacts_host_path = "${local.perma_disk_mount_path}/artifacts"
  perma_htpasswd_host_path  = "${local.perma_disk_mount_path}/nginx/.htpasswd"
  artifact_registry_host    = "${var.region}-docker.pkg.dev"

  # Use templatefile to avoid defining docker compose in this file.
  docker_compose_yaml = templatefile("${path.module}/docker-compose.yml.tftpl", {
    gateway_image              = local.gateway_image
    gateway_port               = var.gateway_port
    react_image                = local.react_image
    react_port                 = var.react_port
    mlflow_image               = local.mlflow_image
    mlflow_port                = var.mlflow_port
    mlflow_bucket              = google_storage_bucket.mlflow_artifacts.name
    perma_db_host_path         = local.perma_db_host_path
    perma_artifacts_host_path  = local.perma_artifacts_host_path
    perma_htpasswd_host_path   = local.perma_htpasswd_host_path
  })

  nginx_default_conf = templatefile("${path.module}/nginx-default.conf.tftpl", {
    react_port = var.react_port
    mlflow_port = var.mlflow_port
  })
}

# Persistent disk used to store MLflow's state database so it survives instance rebuilds
resource "google_compute_disk" "perma_disk" {
  name = "textflow-perma-disk"
  type = "pd-balanced"
  zone = var.zone
  size = var.perma_disk_size_gb
}

# Runs containers in Elastic Compute and exposes the desired port
resource "google_compute_instance" "default" {
  name                      = "textflow-instance"
  machine_type              = "e2-standard-2"
  allow_stopping_for_update = true

  tags = ["textflow-react"]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu_2204_lts.self_link
    }
  }

  attached_disk {
    source      = google_compute_disk.perma_disk.id
    device_name = google_compute_disk.perma_disk.name
    mode        = "READ_WRITE"
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
    "google-logging-enabled"    = "true"
    "google-monitoring-enabled" = "true"
  }

  metadata_startup_script = templatefile("${path.module}/instance_startup.sh.tftpl", {
    disk_name                  = google_compute_disk.perma_disk.name
    mount_path                 = local.perma_disk_mount_path
    docker_compose             = local.docker_compose_yaml
    artifact_registry_host     = local.artifact_registry_host
    perma_db_host_path         = local.perma_db_host_path
    perma_artifacts_host_path  = local.perma_artifacts_host_path
    perma_htpasswd_host_path   = local.perma_htpasswd_host_path
    nginx_conf                 = local.nginx_default_conf
  })

  depends_on = [
    google_artifact_registry_repository.textflow-react,
    google_artifact_registry_repository.textflow_mlflow,
    google_storage_bucket.mlflow_artifacts,
    google_compute_disk.perma_disk,
  ]
}

# Instance service account used to pull images and publish logs/metrics
resource "google_service_account" "textflow_instance_sa" {
  account_id   = "textflow-instance-sa"
  display_name = "TextFlow Compute Engine"
}

# Dedicated bucket to persist MLflow artifacts
resource "google_storage_bucket" "mlflow_artifacts" {
  name                        = "${var.project_id}-mlflow-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  storage_class               = "STANDARD"

  versioning {
    enabled = true
  }
}

# Grant the instance service account access to read/write MLflow artifacts
resource "google_storage_bucket_iam_member" "mlflow_artifacts_instance_rw" {
  bucket = google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.textflow_instance_sa.email}"
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

# Open the Nginx gateway's port to internet clients
resource "google_compute_firewall" "textflow_gateway" {
  name    = "allow-textflow-gateway"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = [tostring(var.gateway_port)]
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

# Holds the MLflow container images in Artifact Registry
resource "google_artifact_registry_repository" "textflow_mlflow" {
  location      = var.region
  repository_id = "textflow-mlflow"
  description   = "Docker repository for MLflow tracking server images"
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
