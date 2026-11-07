#!/usr/bin/env bash
set -e

# ==============================================================================
# Navidrome Autoplay — Installer
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "======================================================================"
echo "      Navidrome Autoplay — Installation"
echo "======================================================================"
echo -e "${RESET}"

# 1. Root check
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}[ERROR] Please run as root (or with sudo).${RESET}"
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# Determine service user: if installed inside /root or /home, systemd's ProtectHome
# or 0700 home directory permissions block unprivileged users, so only drop privileges
# when installed in a system path like /opt or /srv.
if [[ "$APP_DIR" == /root/* || "$APP_DIR" == /home/* ]]; then
    SERVICE_USER="root"
    SERVICE_GROUP="root"
    PROTECT_HOME="false"
else
    SERVICE_USER="navidrome"
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /usr/sbin/nologin navidrome 2>/dev/null || true
    fi
    SERVICE_GROUP="$(id -gn "$SERVICE_USER" 2>/dev/null || echo "$SERVICE_USER")"
    PROTECT_HOME="true"
fi

# 2. System dependencies (do not swallow errors with || true)
echo -e "${YELLOW}[1/5] Checking system packages (python3, ffmpeg, build tools)...${RESET}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq curl python3 python3-dev build-essential ffmpeg libaubio-dev libaubio5
elif command -v dnf &> /dev/null; then
    dnf install -y -q curl python3 python3-devel gcc ffmpeg
elif command -v pacman &> /dev/null; then
    pacman -Sy --noconfirm curl python base-devel ffmpeg aubio
fi

# 3. Install uv (if not present) & sync project environment
echo -e "${YELLOW}[2/5] Setting up uv and syncing Python dependencies...${RESET}"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
fi
UV_BIN="$(command -v uv || echo /usr/local/bin/uv)"
VENV_DIR="$APP_DIR/.venv"
"$UV_BIN" sync --no-dev --project "$APP_DIR"

# Ensure permissions for the service user on APP_DIR
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$APP_DIR"
chmod 755 "$APP_DIR"

# 4. Initial database check & feature extraction (pass APP_DIR via environment variable)
echo -e "${YELLOW}[3/5] Inspecting Navidrome database...${RESET}"
APP_DIR="$APP_DIR" "$VENV_DIR/bin/python" -c "
import os, sys
app_dir = os.environ.get('APP_DIR')
if app_dir:
    sys.path.insert(0, app_dir)
import config
print(f'Detected Navidrome DB at: {config.PRIMARY_DB_PATH}')
"

# 5. Setup systemd service
echo -e "${YELLOW}[4/5] Setting up systemd service (navidrome-autoplay.service)...${RESET}"
SERVICE_FILE="/etc/systemd/system/navidrome-autoplay.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Navidrome Autoplay Recommendation Service
After=network.target navidrome.service
Wants=navidrome.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/python $APP_DIR/server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AUTOPLAY_PORT=8001
Environment=AUTOPLAY_HOST=127.0.0.1

# Sandboxing & Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=$PROTECT_HOME
ReadWritePaths=$APP_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

if command -v systemctl &>/dev/null && [ -d /run/systemd/system ]; then
    systemctl daemon-reload
    systemctl enable --now navidrome-autoplay || true
fi

# 6. Verify health
echo -e "${YELLOW}[5/5] Verifying API health (waiting up to 15s for initial graph build)...${RESET}"
STATUS=""
for i in {1..15}; do
  STATUS=$(curl -s http://127.0.0.1:8001/api/status || curl -s http://127.0.0.1:8000/api/status || echo "")
  if [[ "$STATUS" == *"online"* ]]; then
    break
  fi
  sleep 1
done

if [[ "$STATUS" == *"online"* ]]; then
  echo -e "${GREEN}${BOLD}✓ Service is ONLINE and listening on ports 8000 & 8001!${RESET}"
else
  echo -e "${YELLOW}[!] Service started. Check logs: journalctl -u navidrome-autoplay -f${RESET}"
fi

echo ""
echo -e "${BOLD}${GREEN}Installation Complete!${RESET}"
echo "----------------------------------------------------------------------"
echo -e "To scan all audio files for exact BPM and Camelot keys, run:"
echo -e "  ${CYAN}uv run analyzer.py${RESET}"
echo ""
echo -e "In Navidrome, set the AudioMuse-AI plugin API URL to:"
echo -e "  ${CYAN}http://127.0.0.1:8001${RESET} (or http://127.0.0.1:8000)"
echo "----------------------------------------------------------------------"
