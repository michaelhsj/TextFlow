#!/bin/bash
set -euxo pipefail

# Capture logs from the GPU runner provisioning for troubleshooting.
exec > >(tee /var/log/startup-gpu-runner.log) 2>&1

# Suppress interactive prompts during package installation.
export DEBIAN_FRONTEND=noninteractive

# Install Docker, Python, and system utilities required to execute GPU jobs.
apt-get update
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  python3 \
  python3-pip \
  software-properties-common \
  docker.io \
  docker-compose

# Enable and start the Docker daemon so containers can run immediately.
systemctl enable docker
systemctl start docker

# Install NVIDIA GPU drivers if they are not already present.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  curl -s -o /tmp/install_gpu_driver.py https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/stable/linux/install_gpu_driver.py
  python3 /tmp/install_gpu_driver.py
fi

# Install the NVIDIA container toolkit when missing to allow Docker GPU passthrough.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# Create a helper script to run GPU-enabled container jobs with optional auth.
cat <<'SCRIPT' >/usr/local/bin/run_gpu_container_job.sh
#!/usr/bin/env bash
set -euo pipefail

IMAGE=""
GPUS="all"
declare -a ENV_OPTS
declare -a VOL_OPTS
WORKDIR=""
CONTAINER_NAME=""
declare -a COMMAND

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --env)
      ENV_OPTS+=("$2")
      shift 2
      ;;
    --volume)
      VOL_OPTS+=("$2")
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "--image is required" >&2
  exit 1
fi

REGISTRY="$${IMAGE%%/*}"
if [[ "$REGISTRY" != "$IMAGE" ]]; then
  case "$REGISTRY" in
    *.pkg.dev|*.gcr.io|gcr.io)
      TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
        python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
      printf '%s' "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY" >/dev/null
      ;;
  esac
fi

DOCKER_CMD=(docker run --rm --gpus "$GPUS")

for entry in "$${ENV_OPTS[@]}"; do
  DOCKER_CMD+=(--env "$entry")
done

for entry in "$${VOL_OPTS[@]}"; do
  DOCKER_CMD+=(--volume "$entry")
done

if [[ -n "$WORKDIR" ]]; then
  DOCKER_CMD+=(--workdir "$WORKDIR")
fi

if [[ -n "$CONTAINER_NAME" ]]; then
  DOCKER_CMD+=(--name "$CONTAINER_NAME")
fi

DOCKER_CMD+=("$IMAGE")

if [[ $${#COMMAND[@]} -gt 0 ]]; then
  DOCKER_CMD+=("$${COMMAND[@]}")
fi

"$${DOCKER_CMD[@]}"
SCRIPT

chmod +x /usr/local/bin/run_gpu_container_job.sh
