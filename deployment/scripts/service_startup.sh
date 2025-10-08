#!/bin/bash
set -euo pipefail

# Mirror stdout/stderr to the instance startup log for debugging.
exec > >(tee /var/log/startup-instance.log) 2>&1

# Define persistent disk, mount, and deployment paths provided by metadata.
DISK_DEV="/dev/disk/by-id/google-${disk_name}"
MOUNT_POINT="${mount_path}"
DB_PATH="${perma_db_host_path}"
ARTIFACTS_PATH="${perma_artifacts_host_path}"
DAGSTER_HOME_PATH="${perma_dagster_home_host_path}"
COMPOSE_ROOT="/opt/textflow"
GATEWAY_DIR="$COMPOSE_ROOT/gateway"
COMPOSE_FILE="$COMPOSE_ROOT/docker-compose.yml"
NGINX_CONF="$GATEWAY_DIR/default.conf"
HTPASSWD_FILE="${perma_htpasswd_host_path}"
AUTH_DIR="$(dirname "$HTPASSWD_FILE")"

# Prevent Debian tools from prompting during package installs.
export DEBIAN_FRONTEND=noninteractive

# Install base tooling required before adding the Docker repository.
apt-get update
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  python3 \
  python3-pip \
  apache2-utils

# Configure Docker's official repository for the LTS image.
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --batch --yes --no-tty --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

# Install Docker Engine along with Buildx and Compose v2.
apt-get update
apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-compose-plugin

# Ensure Docker is enabled at boot and running now.
systemctl enable docker
systemctl start docker

# Wait for the persistent disk device to appear before continuing.
for _ in $(seq 1 15); do
  if [[ -b "$DISK_DEV" ]]; then
    break
  fi
  sleep 2
done

# Abort if the disk never became available.
if [[ ! -b "$DISK_DEV" ]]; then
  echo "Persistent disk $DISK_DEV not found" >&2
  exit 1
fi

# Format the disk if it does not already contain a filesystem.
if ! blkid "$DISK_DEV" >/dev/null 2>&1; then
  mkfs.ext4 -F "$DISK_DEV"
fi

# Capture the disk UUID for fstab entries.
UUID=$(blkid -s UUID -o value "$DISK_DEV")

# Create the mount point and ensure it exists before mounting.
mkdir -p "$MOUNT_POINT"

# Persist the mount configuration if it is not already recorded.
if ! grep -qs "$UUID" /etc/fstab; then
  echo "UUID=$UUID $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
fi

# Mount the persistent disk if it is not mounted yet.
if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$MOUNT_POINT"
fi

# Prepare data directories with sane permissions on the mounted disk.
mkdir -p "$DB_PATH" "$ARTIFACTS_PATH" "$DAGSTER_HOME_PATH"
chmod 755 "$MOUNT_POINT" "$DB_PATH" "$ARTIFACTS_PATH" "$DAGSTER_HOME_PATH"

# Lay out directories for docker compose, gateway, and auth material.
mkdir -p "$COMPOSE_ROOT" "$GATEWAY_DIR" "$AUTH_DIR"
touch "$HTPASSWD_FILE"
chmod 644 "$HTPASSWD_FILE"
echo "Nginx basic auth enabled for /mlflow and /dagster; add users with 'sudo htpasswd $HTPASSWD_FILE <user>'" >> /var/log/startup-instance.log

# Write docker compose definition supplied from instance metadata.
cat <<'COMPOSE' > "$COMPOSE_FILE"
${docker_compose}
COMPOSE

# Generate the Nginx configuration from the rendered template.
cat <<'NGINX' > "$NGINX_CONF"
${nginx_conf}
NGINX

# Authenticate to Artifact Registry if a host is provided.
if [[ -n "${artifact_registry_host}" ]]; then
  TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
  printf '%s' "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://${artifact_registry_host}" || true
fi

# Pull required container images prior to starting services.
docker compose -f "$COMPOSE_FILE" pull

# Launch the docker compose stack in detached mode, cleaning old containers.
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
