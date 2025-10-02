# GPU job runner resources live here to keep the main instance configuration focused.

# Holds GPU job container images in Artifact Registry (created on demand)
resource "google_artifact_registry_repository" "textflow_jobs" {
  count         = var.enable_gpu_job_runner ? 1 : 0
  location      = var.region
  repository_id = "textflow-jobs"
  description   = "Docker repository for GPU job workloads"
  format        = "DOCKER"
}

# Service account backing the GPU job runner instance
resource "google_service_account" "gpu_job_runner_sa" {
  count        = var.enable_gpu_job_runner ? 1 : 0
  account_id   = "textflow-gpu-runner-sa"
  display_name = "TextFlow GPU Job Runner"
}

# Allow the GPU runner to read Artifact Registry images
resource "google_project_iam_member" "gpu_runner_artifact_registry" {
  count   = var.enable_gpu_job_runner ? 1 : 0
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gpu_job_runner_sa[count.index].email}"
}

# Allow the GPU runner to read from project buckets when jobs need dataset pulls
resource "google_project_iam_member" "gpu_runner_storage_viewer" {
  count   = var.enable_gpu_job_runner ? 1 : 0
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.gpu_job_runner_sa[count.index].email}"
}

# Permit GPU runner logs and metrics to land in Cloud Logging/Monitoring
resource "google_project_iam_member" "gpu_runner_logging" {
  count   = var.enable_gpu_job_runner ? 1 : 0
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gpu_job_runner_sa[count.index].email}"
}

resource "google_project_iam_member" "gpu_runner_monitoring" {
  count   = var.enable_gpu_job_runner ? 1 : 0
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gpu_job_runner_sa[count.index].email}"
}

# Standalone GPU-enabled instance for ad-hoc containerized jobs
resource "google_compute_instance" "gpu_job_runner" {
  count                     = var.enable_gpu_job_runner ? 1 : 0
  name                      = "textflow-gpu-runner"
  machine_type              = var.gpu_job_machine_type
  allow_stopping_for_update = true

  tags = var.gpu_runner_tags

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu_2204_lts.self_link
      size  = var.gpu_boot_disk_size_gb
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = google_service_account.gpu_job_runner_sa[count.index].email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata_startup_script = file("${path.module}/gpu_runner_startup.sh")
}
