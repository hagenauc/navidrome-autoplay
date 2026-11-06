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

# 2. System dependencies
echo -e "${YELLOW}[1/5] Checking system packages (python3, ffmpeg, build tools)...${RESET}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv ffmpeg libaubio-dev libaubio5 > /dev/null 2>&1 || true
elif command -v dnf &> /dev/null; then
    dnf install -y -q python3 python3-pip ffmpeg > /dev/null 2>&1 || true
elif command -v pacman &> /dev/null; then
    pacman -Sy --noconfirm python python-pip ffmpeg aubio > /dev/null 2>&1 || true
fi

# 3. Python dependencies
echo -e "${YELLOW}[2/5] Installing Python dependencies...${RESET}"
pip3 install -r requirements.txt --quiet --break-system-packages 2>/dev/null || pip3 install -r requirements.txt --quiet

# 4. Initial database check & feature extraction
echo -e "${YELLOW}[3/5] Inspecting Navidrome database...${RESET}"
python3 -c "
import os, sys
sys.path.insert(0, '$APP_DIR')
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
User=root
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/python3 $APP_DIR/server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AUTOPLAY_PORT=8001
Environment=AUTOPLAY_HOST=0.0.0.0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now navidrome-autoplay

# 6. Verify health
echo -e "${YELLOW}[5/5] Verifying API health...${RESET}"
sleep 2

STATUS=$(curl -s http://127.0.0.1:8001/api/status || curl -s http://127.0.0.1:8000/api/status || echo "")

if [[ "$STATUS" == *"online"* ]]; then
  echo -e "${GREEN}${BOLD}✓ Service is ONLINE and listening on ports 8000 & 8001!${RESET}"
else
  echo -e "${YELLOW}[!] Service started. Check logs: journalctl -u navidrome-autoplay -f${RESET}"
fi

echo ""
echo -e "${BOLD}${GREEN}Installation Complete!${RESET}"
echo "----------------------------------------------------------------------"
echo -e "To scan all audio files for exact BPM and Camelot keys, run:"
echo -e "  ${CYAN}python3 analyzer.py${RESET}"
echo ""
echo -e "In Navidrome, set the AudioMuse-AI plugin API URL to:"
echo -e "  ${CYAN}http://127.0.0.1:8001${RESET} (or http://127.0.0.1:8000)"
echo "----------------------------------------------------------------------"
