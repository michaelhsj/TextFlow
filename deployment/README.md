# deployment
Terraform, docker, and anything else that is used for cloud workflows.

- `terraform/` now includes an optional GPU-backed job runner VM. Enable it with `TF_VAR_enable_gpu_job_runner=true` and use `deployment/scripts/submit_gpu_job.py` to launch containerized workloads on the instance.
