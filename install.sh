#!/usr/bin/env bash
set -e

# ==============================================================================
# Navidrome Autoplay — Installer (Least-Privilege / Rootless-Friendly)
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

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# 1. Check system tools (only invoke a package manager if something is actually missing)
echo -e "${YELLOW}[1/5] Checking required system tools (python3, ffmpeg, C compiler)...${RESET}"
MISSING_TOOLS=0
for tool in python3 ffmpeg cc; do
    if ! command -v "$tool" &>/dev/null; then
        MISSING_TOOLS=1
        break
    fi
done

if [ "$MISSING_TOOLS" -eq 0 ]; then
    echo "All required system tools (python3, ffmpeg, cc) are already installed."
else
    echo "Installing missing system packages..."
    SUDO=""
    if [ "$EUID" -ne 0 ]; then
        if command -v sudo &>/dev/null; then
            SUDO="sudo"
        else
            echo -e "${RED}[ERROR] Missing system dependencies (python3, ffmpeg, or C compiler) and neither root nor sudo is available.${RESET}"
            echo "Please install ffmpeg, python3-dev, and build-essential via your package manager and re-run ./install.sh."
            exit 1
        fi
    fi

    if command -v apt-get &>/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq curl python3 python3-dev build-essential ffmpeg libaubio-dev libaubio5
    elif command -v dnf &>/dev/null; then
        $SUDO dnf install -y -q curl python3 python3-devel gcc ffmpeg
    elif command -v pacman &>/dev/null; then
        $SUDO pacman -Sy --noconfirm curl python base-devel ffmpeg aubio
    else
        echo -e "${RED}[ERROR] Unsupported package manager. Please install python3, ffmpeg, and a C compiler manually.${RESET}"
        exit 1
    fi
fi

# 2. Install uv (into ~/.local/bin, no root required) & sync project environment
echo -e "${YELLOW}[2/5] Setting up uv and syncing Python dependencies...${RESET}"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
VENV_DIR="$APP_DIR/.venv"
"$UV_BIN" sync --no-dev --project "$APP_DIR"

# 3. Initial database check (pass APP_DIR via environment variable)
echo -e "${YELLOW}[3/5] Inspecting Navidrome database...${RESET}"
APP_DIR="$APP_DIR" "$VENV_DIR/bin/python" -c "
import os, sys
app_dir = os.environ.get('APP_DIR')
if app_dir:
    sys.path.insert(0, app_dir)
import config
print(f'Detected Navidrome DB at: {config.PRIMARY_DB_PATH}')
if not os.access(config.PRIMARY_DB_PATH, os.R_OK):
    print(f'Warning: {config.PRIMARY_DB_PATH} is not readable by the current user ({os.getuid()}).')
"

# 4. Setup service (rootless user service or service.sh when non-root; system service when root)
echo -e "${YELLOW}[4/5] Starting Navidrome Autoplay service...${RESET}"

if [ "$EUID" -ne 0 ]; then
    # Non-root installation: prefer systemd --user if a user session bus is active, otherwise use service.sh
    if command -v systemctl &>/dev/null && systemctl --user status >/dev/null 2>&1; then
        USER_UNIT_DIR="$HOME/.config/systemd/user"
        mkdir -p "$USER_UNIT_DIR"
        SERVICE_FILE="$USER_UNIT_DIR/navidrome-autoplay.service"
        cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Navidrome Autoplay Recommendation Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/python $APP_DIR/server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=AUTOPLAY_PORT=8001
Environment=AUTOPLAY_HOST=127.0.0.1
NoNewPrivileges=true

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now navidrome-autoplay
        MANAGE_HINT="systemctl --user status navidrome-autoplay"
    else
        "$APP_DIR/service.sh" restart
        MANAGE_HINT="$APP_DIR/service.sh status"
    fi
else
    # Root installation (e.g. LXC container or system-wide /opt install)
    if [[ "$APP_DIR" == /root/* || "$APP_DIR" == /home/* ]]; then
        SERVICE_USER="$(stat -c '%U' "$APP_DIR" 2>/dev/null || echo root)"
        SERVICE_GROUP="$(stat -c '%G' "$APP_DIR" 2>/dev/null || echo root)"
        PROTECT_HOME="false"
    else
        if id "navidrome" &>/dev/null; then
            SERVICE_USER="navidrome"
            SERVICE_GROUP="$(id -gn navidrome 2>/dev/null || echo navidrome)"
        else
            SERVICE_USER="$(stat -c '%U' "$APP_DIR" 2>/dev/null || echo root)"
            SERVICE_GROUP="$(stat -c '%G' "$APP_DIR" 2>/dev/null || echo root)"
        fi
        chown -R "$SERVICE_USER:$SERVICE_GROUP" "$APP_DIR"
        PROTECT_HOME="true"
    fi

    if command -v systemctl &>/dev/null && [ -d /run/systemd/system ]; then
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
        systemctl daemon-reload
        systemctl enable --now navidrome-autoplay
        MANAGE_HINT="systemctl status navidrome-autoplay"
    else
        "$APP_DIR/service.sh" restart
        MANAGE_HINT="$APP_DIR/service.sh status"
    fi
fi

# 5. Verify health
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
  echo -e "${GREEN}${BOLD}✓ Service is ONLINE and listening on 127.0.0.1 ports 8000 & 8001!${RESET}"
else
  echo -e "${YELLOW}[!] Service started. Check status with: $MANAGE_HINT${RESET}"
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
