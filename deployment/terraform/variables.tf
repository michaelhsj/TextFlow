variable "project_id" {
  description = "The project ID to deploy to."
  type        = string
}

variable "region" {
  description = "The region to deploy to."
  type        = string
  default     = "northamerica-northeast1"
}

variable "zone" {
  description = "The zone to deploy to."
  type        = string
  default     = "northamerica-northeast1-a"
}

variable "react_port" {
  description = "Port exposed by the React container."
  type        = number
  default     = 3000
}
