variable "project_id" {
  description = "The project ID to deploy to."
  type        = string
  default     = "double-goal-473318-a4"
}

variable "region" {
  description = "The region to deploy to."
  type        = string
  default     = "northamerica-northeast1"
}

variable "service_zone" {
  description = "Zone for the primary TextFlow service instance and its disk."
  type        = string
  default     = "northamerica-northeast1-c"
}

variable "gpu_region" {
  description = "Region used for GPU runner networking resources (defaults to the region portion of gpu_zone)."
  type        = string
  default     = "northamerica-northeast2"
}

variable "gpu_zone" {
  description = "Zone for the optional GPU job runner instance."
  type        = string
  default     = "northamerica-northeast2-a"
}

variable "gpu_job_machine_type" {
  description = "Machine type for the GPU job runner instance."
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "GPU accelerator type to attach to the job runner instance."
  type        = string
  default     = "nvidia-l4"
}

variable "react_port" {
  description = "Port exposed by the React container."
  type        = number
  default     = 3000
}

variable "mlflow_port" {
  description = "Port exposed by the MLflow tracking server."
  type        = number
  default     = 5000
}

variable "dagster_port" {
  description = "Port exposed by the Dagster webserver."
  type        = number
  default     = 3000
}

variable "perma_disk_size_gb" {
  description = "Size (GB) of the persistent disk that stores MLflow state."
  type        = number
  default     = 20
}

variable "gateway_port" {
  description = "Public port exposed by the Nginx gateway."
  type        = number
  default     = 80
}

variable "enable_gpu_job_runner" {
  description = "Provision a GPU-backed instance for running ad-hoc container jobs."
  type        = bool
  default     = true
}



variable "gpu_count" {
  description = "Number of GPU accelerators to attach to the job runner instance."
  type        = number
  default     = 1
}

variable "gpu_boot_disk_size_gb" {
  description = "Boot disk size (GB) for the GPU job runner instance."
  type        = number
  default     = 200
}

variable "gpu_runner_docker_url" {
  description = "Docker daemon endpoint exposed by the GPU job runner."
  type        = string
  default     = "tcp://textflow-gpu-runner:2375"
}

variable "gpu_runner_tags" {
  description = "Network tags to apply to the GPU job runner instance."
  type        = list(string)
  default     = ["textflow-gpu-runner"]
}

variable "cloudflare_api_token" {
  description = "API token used for authenticating with Cloudflare."
  type        = string
  sensitive   = true

  validation {
    condition     = trimspace(var.cloudflare_api_token) != ""
    error_message = "Set TF_VAR_cloudflare_api_token (or pass -var cloudflare_api_token=...) before running Terraform."
  }
}
