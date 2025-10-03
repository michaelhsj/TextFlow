#!/bin/bash
set -euo pipefail

exec > >(tee /var/log/startup-instance.log) 2>&1

DISK_DEV="/dev/disk/by-id/google-${disk_name}"
MOUNT_POINT="${mount_path}"
DB_PATH="${perma_db_host_path}"
ARTIFACTS_PATH="${perma_artifacts_host_path}"
COMPOSE_ROOT="/opt/textflow"
GATEWAY_DIR="$COMPOSE_ROOT/gateway"
COMPOSE_FILE="$COMPOSE_ROOT/docker-compose.yml"
NGINX_CONF="$GATEWAY_DIR/default.conf"
HTPASSWD_FILE="${perma_htpasswd_host_path}"
AUTH_DIR="$(dirname "$HTPASSWD_FILE")"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  python3 \
  python3-pip \
  docker.io \
  docker-compose \
  apache2-utils

systemctl enable docker
systemctl start docker

for _ in $(seq 1 15); do
  if [[ -b "$DISK_DEV" ]]; then
    break
  fi
  sleep 2
done

if [[ ! -b "$DISK_DEV" ]]; then
  echo "Persistent disk $DISK_DEV not found" >&2
  exit 1
fi

if ! blkid "$DISK_DEV" >/dev/null 2>&1; then
  mkfs.ext4 -F "$DISK_DEV"
fi

UUID=$(blkid -s UUID -o value "$DISK_DEV")

mkdir -p "$MOUNT_POINT"

if ! grep -qs "$UUID" /etc/fstab; then
  echo "UUID=$UUID $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
fi

if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$MOUNT_POINT"
fi

mkdir -p "$DB_PATH" "$ARTIFACTS_PATH"
chmod 755 "$MOUNT_POINT" "$DB_PATH" "$ARTIFACTS_PATH"

mkdir -p "$COMPOSE_ROOT" "$GATEWAY_DIR" "$AUTH_DIR"
touch "$HTPASSWD_FILE"
chmod 644 "$HTPASSWD_FILE"
echo "Nginx basic auth enabled for /mlflow; add users with 'sudo htpasswd $HTPASSWD_FILE <user>'" >> /var/log/startup-instance.log

cat <<'COMPOSE' > "$COMPOSE_FILE"
${docker_compose}
COMPOSE

cat <<'NGINX' > "$NGINX_CONF"
${nginx_conf}
NGINX

if [[ -n "${artifact_registry_host}" ]]; then
  TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
  printf '%s' "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://${artifact_registry_host}" || true
fi

docker-compose -f "$COMPOSE_FILE" pull

docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans
