#!/usr/bin/env bash
# Install the WanGP Agent API server as a systemd user unit.
#
# Usage:
#   ./deploy/install-systemd-unit.sh                   # interactive
#   WAN2GP_TOKEN=secret ./deploy/install-systemd-unit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${SCRIPT_DIR}/wan2gp-api.service"
DEST_DIR="${HOME}/.config/systemd/user"
DEST="${DEST_DIR}/wan2gp-api.service"

if [[ ! -f "${SRC}" ]]; then
  echo "error: ${SRC} missing" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

TOKEN="${WAN2GP_TOKEN:-}"
if [[ -z "${TOKEN}" ]]; then
  read -r -p "Bearer token (leave blank for no auth): " TOKEN || true
fi

CORS="${WAN2GP_CORS_ORIGINS:-}"
if [[ -z "${CORS}" ]] && [[ -t 0 ]]; then
  read -r -p "CORS origins (comma-separated, '*', or blank to disable): " CORS || true
fi

# Substitute repo path, token, and CORS origins into a copy of the unit file.
python3 - "$SRC" "$DEST" "$REPO_DIR" "$TOKEN" "$CORS" <<'PY'
import sys
src, dest, repo, token, cors = sys.argv[1:6]
text = open(src).read()
text = text.replace("/media/peter/AI/Wan2GP", repo)
text = text.replace("Environment=WAN2GP_TOKEN=",
                    f"Environment=WAN2GP_TOKEN={token}")
text = text.replace("Environment=WAN2GP_CORS_ORIGINS=",
                    f"Environment=WAN2GP_CORS_ORIGINS={cors}")
open(dest, "w").write(text)
PY

systemctl --user daemon-reload
systemctl --user enable --now wan2gp-api.service

if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$USER" 2>/dev/null || true
fi

sleep 2
systemctl --user --no-pager status wan2gp-api.service || true

cat <<EOF

Installed at: ${DEST}

  Logs:    journalctl --user -u wan2gp-api -f
  Stop:    systemctl --user stop wan2gp-api
  Restart: systemctl --user restart wan2gp-api
  Status:  systemctl --user status wan2gp-api

Health check (LAN):  curl http://\$(hostname -I | awk '{print \$1}'):8100/api/health
EOF
