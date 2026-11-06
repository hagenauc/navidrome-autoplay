#!/usr/bin/env bash
#
# Navidrome Autoplay Service Manager
# High-performance, session-based music recommendation engine for Navidrome
#

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$DIR/autoplay.pid"
LOG_FILE="$DIR/autoplay.log"
PYTHON_BIN="/usr/bin/python3"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Navidrome Autoplay is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    echo "Starting Navidrome Autoplay on port 8001..."
    nohup "$PYTHON_BIN" "$DIR/server.py" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
    sleep 2

    if kill -0 "$PID" 2>/dev/null; then
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
        echo "Navidrome Autoplay is RUNNING (PID: $PID)"
        if command -v curl >/dev/null 2>&1; then
            echo "Health status:"
            curl -s http://127.0.0.1:8001/health || echo "Could not reach health endpoint"
            echo ""
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
