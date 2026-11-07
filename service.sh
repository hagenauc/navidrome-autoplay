#!/usr/bin/env bash
#
# Navidrome Autoplay Service Manager
# High-performance, session-based music recommendation engine for Navidrome
#

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$DIR/autoplay.pid"
LOG_FILE="$DIR/autoplay.log"

if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || echo /usr/bin/python3)}"
fi

is_autoplay_process() {
    local pid="$1"
    if [ -f "/proc/$pid/cmdline" ]; then
        if grep -q -E "server\.py|gunicorn|navidrome-autoplay" "/proc/$pid/cmdline" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start() {
    if [ -f "$PID_FILE" ]; then
        PID="$(cat "$PID_FILE")"
        if kill -0 "$PID" 2>/dev/null && is_autoplay_process "$PID"; then
            echo "Navidrome Autoplay is already running (PID: $PID)"
            return 0
        fi
    fi

    echo "Starting Navidrome Autoplay on port 8001..."
    nohup "$PYTHON_BIN" "$DIR/server.py" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
    sleep 2

    if kill -0 "$PID" 2>/dev/null && is_autoplay_process "$PID"; then
        echo "Navidrome Autoplay started successfully (PID: $PID)"
        echo "Logs: $LOG_FILE"
    else
        echo "Failed to start Navidrome Autoplay. Recent logs:"
        tail -n 20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID="$(cat "$PID_FILE")"
        if kill -0 "$PID" 2>/dev/null; then
            if is_autoplay_process "$PID"; then
                echo "Stopping Navidrome Autoplay (PID: $PID)..."
                kill "$PID"
                for i in {1..10}; do
                    if ! kill -0 "$PID" 2>/dev/null; then
                        break
                    fi
                    sleep 0.5
                done
                if kill -0 "$PID" 2>/dev/null; then
                    echo "Force killing PID: $PID"
                    kill -9 "$PID"
                fi
            else
                echo "Warning: PID $PID does not match Navidrome Autoplay process identity. Skipping kill to protect recycled PID."
            fi
        fi
        rm -f "$PID_FILE"
        echo "Navidrome Autoplay stopped."
    else
        echo "Navidrome Autoplay is not running (no PID file)."
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        PID="$(cat "$PID_FILE")"
        if is_autoplay_process "$PID"; then
            echo "Navidrome Autoplay is RUNNING (PID: $PID)"
            if command -v curl >/dev/null 2>&1; then
                echo "Health status:"
                curl -s http://127.0.0.1:8001/health || echo "Could not reach health endpoint"
                echo ""
            fi
        else
            echo "PID $PID exists but does not match Navidrome Autoplay command line."
        fi
    else
        echo "Navidrome Autoplay is STOPPED"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
