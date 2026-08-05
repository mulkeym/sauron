#!/usr/bin/env bash
set -euo pipefail

container_name="open-webui"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_file=${OPENWEBUI_ENV_FILE:-"$script_dir/.openwebui.env"}

if [[ ! -f $env_file ]]; then
  printf 'Open WebUI environment file not found: %s\n' "$env_file" >&2
  exit 1
fi

if docker container inspect "$container_name" >/dev/null 2>&1; then
  docker container rm --force "$container_name" >/dev/null
fi

docker run \
  --detach \
  --name "$container_name" \
  --restart unless-stopped \
  --publish 3000:8080 \
  --volume open-webui:/app/backend/data \
  --env-file "$env_file" \
  ghcr.io/open-webui/open-webui:main
