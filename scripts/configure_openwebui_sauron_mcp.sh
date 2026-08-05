#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Configure Sauron's native MCP server in an Open WebUI v0.11+ Docker container.

Usage:
  SAURON_API_KEY=... scripts/configure_openwebui_sauron_mcp.sh [options]

Options:
  --container NAME   Open WebUI container name (default: open-webui)
  --url URL          Sauron MCP URL (default: http://host.docker.internal:8880/mcp)
  --user USER        Open WebUI user name or email to receive --groups
  --groups CSV       Comma-separated Sauron ACL group names to create and assign
  --id ID            MCP connection ID (default: sauron)
  --name NAME        Display name (default: Sauron)
  --help             Show this help

SAURON_API_KEY is read from the environment. If it is unset and stdin is a
terminal, the script prompts without echo. The key is never passed as a command
line argument.

Open WebUI must be launched with:
  ENABLE_FORWARD_USER_INFO_HEADERS=true
  FORWARD_USER_INFO_HEADER_JWT_SECRET=<same secret as Sauron's MCP_OPENWEBUI_JWT_SECRET>
  WEBUI_SECRET_KEY=<persistent secret>
EOF
}

container_name="open-webui"
mcp_url="http://host.docker.internal:8880/mcp"
connection_id="sauron"
connection_name="Sauron"
target_user=""
group_csv=""

while (($#)); do
  case "$1" in
    --container) container_name=${2:?missing value for --container}; shift 2 ;;
    --url) mcp_url=${2:?missing value for --url}; shift 2 ;;
    --user) target_user=${2:?missing value for --user}; shift 2 ;;
    --groups) group_csv=${2:?missing value for --groups}; shift 2 ;;
    --id) connection_id=${2:?missing value for --id}; shift 2 ;;
    --name) connection_name=${2:?missing value for --name}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! docker container inspect "$container_name" >/dev/null 2>&1; then
  printf 'Open WebUI container not found: %s\n' "$container_name" >&2
  exit 1
fi

if [[ -z ${SAURON_API_KEY:-} ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p 'Sauron API key: ' SAURON_API_KEY
    printf '\n'
  else
    printf 'Set SAURON_API_KEY or run this script from a terminal.\n' >&2
    exit 1
  fi
fi

if [[ -z $target_user && -n $group_csv ]]; then
  printf -- '--groups requires --user.\n' >&2
  exit 2
fi

forwarding=$(docker exec "$container_name" sh -c 'printf %s "${ENABLE_FORWARD_USER_INFO_HEADERS:-}"')
identity_secret=$(docker exec "$container_name" sh -c 'if [ -n "${FORWARD_USER_INFO_HEADER_JWT_SECRET:-}" ]; then printf set; fi')
webui_secret=$(docker exec "$container_name" sh -c 'if [ -n "${WEBUI_SECRET_KEY:-}" ]; then printf set; fi')

if [[ ${forwarding,,} != true || $identity_secret != set || $webui_secret != set ]]; then
  printf '%s\n' \
    'Open WebUI identity forwarding is not fully configured.' \
    'Set ENABLE_FORWARD_USER_INFO_HEADERS=true, FORWARD_USER_INFO_HEADER_JWT_SECRET,' \
    'and a persistent WEBUI_SECRET_KEY, then recreate the container.' >&2
  exit 1
fi

docker exec \
  --interactive \
  -e SETUP_MCP_URL="$mcp_url" \
  -e SETUP_MCP_ID="$connection_id" \
  -e SETUP_MCP_NAME="$connection_name" \
  -e SETUP_MCP_USER="$target_user" \
  -e SETUP_MCP_GROUPS="$group_csv" \
  -e SETUP_SAURON_API_KEY="$SAURON_API_KEY" \
  "$container_name" python - <<'PY'
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

db_path = Path('/app/backend/data/webui.db')
if not db_path.is_file():
    raise SystemExit(f'Open WebUI database not found: {db_path}')

source = sqlite3.connect(db_path)
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
backup_path = db_path.with_name(f'webui.db.pre-sauron-mcp-{stamp}')
backup = sqlite3.connect(backup_path)
source.backup(backup)
backup.close()

row = source.execute(
    "SELECT value FROM config WHERE key = 'tool_server.connections'"
).fetchone()
if row is None:
    raise SystemExit(
        'Unsupported Open WebUI configuration schema; this script requires v0.11+'
    )

connections = json.loads(row[0]) if isinstance(row[0], str) else row[0]
connection_id = os.environ['SETUP_MCP_ID']
mcp_url = os.environ['SETUP_MCP_URL'].rstrip('/')

# Preserve unrelated connections. Re-running replaces only this Sauron entry.
connections = [
    item for item in connections
    if not (
        item.get('type') == 'mcp'
        and (
            (item.get('info') or {}).get('id') == connection_id
            or item.get('url', '').rstrip('/') == mcp_url
        )
    )
]
connections.append({
    'type': 'mcp',
    'url': mcp_url,
    'spec_type': 'url',
    'spec': '',
    'path': '',
    'auth_type': 'none',
    'headers': {
        'X-API-Key': os.environ['SETUP_SAURON_API_KEY'],
        'X-Sauron-User-Groups': '{{USER_GROUPS}}',
    },
    'key': '',
    'config': {
        'enable': True,
        'function_name_filter_list': '',
        'access_grants': [],
    },
    'info': {
        'id': connection_id,
        'name': os.environ['SETUP_MCP_NAME'],
        'description': 'Sauron data retrieval MCP',
    },
})

now = int(time.time())
source.execute(
    "UPDATE config SET value = ?, updated_at = ? "
    "WHERE key = 'tool_server.connections'",
    (json.dumps(connections), now),
)

target_user = os.environ.get('SETUP_MCP_USER', '').strip()
groups = list(dict.fromkeys(
    value.strip()
    for value in os.environ.get('SETUP_MCP_GROUPS', '').split(',')
    if value.strip()
))
if target_user and groups:
    user = source.execute(
        'SELECT id FROM user WHERE name = ? OR email = ? LIMIT 1',
        (target_user, target_user),
    ).fetchone()
    if user is None:
        raise SystemExit(f'Open WebUI user not found: {target_user}')
    user_id = user[0]

    for group_name in groups:
        group = source.execute(
            'SELECT id FROM "group" WHERE name = ? LIMIT 1', (group_name,)
        ).fetchone()
        if group is None:
            group_id = str(uuid.uuid4())
            source.execute(
                'INSERT INTO "group" '
                '(id, user_id, name, description, data, meta, permissions, '
                'created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    group_id,
                    user_id,
                    group_name,
                    f'Sauron ACL group: {group_name}',
                    json.dumps({'config': {'share': False}}),
                    None,
                    None,
                    now,
                    now,
                ),
            )
        else:
            group_id = group[0]

        membership = source.execute(
            'SELECT 1 FROM group_member WHERE group_id = ? AND user_id = ?',
            (group_id, user_id),
        ).fetchone()
        if membership is None:
            source.execute(
                'INSERT INTO group_member '
                '(id, group_id, user_id, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), group_id, user_id, now, now),
            )

source.commit()
source.close()

print(f'Configured MCP connection {connection_id!r} -> {mcp_url}')
if groups:
    print(f'Assigned {target_user!r} to {len(groups)} group(s): {", ".join(groups)}')
print(f'Backup: {backup_path}')
print('Restart Open WebUI to load the updated connection.')
PY

docker restart "$container_name" >/dev/null
printf 'Restarted %s. Wait for its health check, then enable %s in a chat.\n' \
  "$container_name" "$connection_name"
