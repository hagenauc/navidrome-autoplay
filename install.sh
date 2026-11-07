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

# Determine service user: prefer 'navidrome', fallback to creating 'navidrome'
SERVICE_USER="navidrome"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /usr/sbin/nologin navidrome 2>/dev/null || true
    SERVICE_USER="navidrome"
fi
SERVICE_GROUP="$(id -gn "$SERVICE_USER" 2>/dev/null || echo "$SERVICE_USER")"

# 2. System dependencies (do not swallow errors with || true)
echo -e "${YELLOW}[1/5] Checking system packages (python3, ffmpeg, build tools)...${RESET}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv ffmpeg libaubio-dev libaubio5
elif command -v dnf &> /dev/null; then
    dnf install -y -q python3 python3-pip python3-virtualenv ffmpeg
elif command -v pacman &> /dev/null; then
    pacman -Sy --noconfirm python python-pip ffmpeg aubio
fi

# 3. Python virtual environment & dependencies
echo -e "${YELLOW}[2/5] Setting up virtual environment and Python dependencies...${RESET}"
VENV_DIR="$APP_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt" --quiet

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

GUNICORN_BIN="$VENV_DIR/bin/gunicorn"
if [ ! -x "$GUNICORN_BIN" ]; then
    GUNICORN_BIN="$(command -v gunicorn || echo "$VENV_DIR/bin/python -m gunicorn")"
fi

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
ExecStart=$GUNICORN_BIN --workers 1 --threads 4 --timeout 30 --bind 127.0.0.1:8001 --bind 127.0.0.1:8000 "server:create_app()"
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AUTOPLAY_PORT=8001
Environment=AUTOPLAY_HOST=127.0.0.1

# Sandboxing & Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

# Also keep the repository copy in sync
cp "$SERVICE_FILE" "$APP_DIR/navidrome-autoplay.service" 2>/dev/null || true

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
echo -e "  ${CYAN}$VENV_DIR/bin/python analyzer.py${RESET}"
echo ""
echo -e "In Navidrome, set the AudioMuse-AI plugin API URL to:"
echo -e "  ${CYAN}http://127.0.0.1:8001${RESET} (or http://127.0.0.1:8000)"
echo "----------------------------------------------------------------------"
