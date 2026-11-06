import os
import sys
import time
import threading
import sqlite3
import subprocess
from flask import Flask, request, jsonify
import config
from engine import AutoplayEngine

app = Flask(__name__)
start_time = time.time()

print("[Server] Initializing AutoplayEngine...")
engine = AutoplayEngine()
print("[Server] AutoplayEngine initialized successfully.")


def auto_feature_sync():
    last_count = 0
    last_analysis_check = time.time()
    last_reload_check = time.time()

    while True:
        time.sleep(15)
        now = time.time()

        # 1. Quick in-memory sync of newly analyzed features
        try:
            count = engine.reload_features()
            if count > 0 and count != last_count:
                last_count = count
                print(f"[Server] Synced {count} analyzed track features into memory.")
        except Exception as e:
            print(f"[Server] Feature sync error: {e}")

        # 2. Check for newly uploaded music every 15 minutes
        if (now - last_analysis_check) > 900:
            last_analysis_check = now
            try:
                conn = config.connect_ro(engine.db_path)
                c = conn.cursor()
                total_in_navidrome = c.execute("SELECT count(*) FROM media_file").fetchone()[0]
                conn.close()

                features_path = config.FEATURES_DB_PATH
                if os.path.exists(features_path):
                    fconn = sqlite3.connect(features_path, timeout=10)
                    fc = fconn.cursor()
                    total_in_features = fc.execute("SELECT count(*) FROM track_features").fetchone()[0]
                    fconn.close()
                else:
                    total_in_features = 0

                if total_in_navidrome > total_in_features:
                    missing = total_in_navidrome - total_in_features
                    print(f"[Server] Detected {missing} unanalyzed tracks in Navidrome. Spawning background analyzer...")
                    subprocess.Popen(
                        [sys.executable, os.path.join(config.APP_DIR, "analyzer.py"), "2"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except Exception as e:
                print(f"[Server] Background check error: {e}")

        # 3. Reload full database every 2 hours to incorporate fresh scrobbles and ratings
        if (now - last_reload_check) > 7200:
            last_reload_check = now
            try:
                print("[Server] Periodic refresh of listening history and graph...")
                engine.load_database(blocking=False)
            except Exception as e:
                print(f"[Server] Periodic reload error: {e}")


sync_thread = threading.Thread(target=auto_feature_sync, daemon=True)
sync_thread.start()


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "navidrome-autoplay",
        "version": "2.0.0",
        "uptime_seconds": round(time.time() - start_time, 1),
        "total_tracks": len(engine.tracks),
        "total_artists": len(engine.artist_tracks),
        "total_artist_connections": engine.artist_connection_count(),
        "database_path": engine.db_path
    })


@app.route("/api/similar_tracks", methods=["GET"])
def similar_tracks():
    item_id = request.args.get("item_id")
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    n = request.args.get("n", default=10, type=int)
    n = max(1, min(n, config.MAX_RESULTS))
    eliminate_dup = request.args.get("eliminate_duplicates", default="true").lower() == "true"
    session_id = request.args.get("session_id", default=None)

    results = engine.get_similar_tracks(
        item_id=item_id,
        n=n,
        eliminate_duplicates=eliminate_dup,
        session_id=session_id
    )

    # Return pure list of tracks as expected by the Navidrome plugin
    return jsonify(results)


@app.route("/api/similar_artists", methods=["GET"])
def similar_artists():
    artist_query = request.args.get("artist")
    if not artist_query:
        return jsonify({"error": "artist parameter is required"}), 400

    count = request.args.get("count", default=request.args.get("artistSimilarCount", default=10, type=int), type=int)
    count = max(1, min(count, config.MAX_RESULTS))
    results = engine.get_similar_artists(artist_identifier=artist_query, count=count)

    return jsonify(results)


@app.route("/api/find_path", methods=["GET"])
def find_path():
    start_id = request.args.get("start_song_id")
    end_id = request.args.get("end_song_id")
    if not start_id or not end_id:
        return jsonify({"error": "start_song_id and end_song_id are required"}), 400

    count = request.args.get("count", default=5, type=int)
    max_steps = request.args.get("max_steps", default=10, type=int)
    max_steps = max(2, min(max_steps, config.MAX_PATH_STEPS))

    result = engine.find_path(
        start_song_id=start_id,
        end_song_id=end_id,
        count=count,
        max_steps=max_steps
    )
    return jsonify(result)


@app.route("/api/analysis_status", methods=["GET"])
def analysis_status():
    analyzed_count = 0
    if os.path.exists(config.FEATURES_DB_PATH):
        try:
            conn = sqlite3.connect(config.FEATURES_DB_PATH, timeout=10)
            c = conn.cursor()
            analyzed_count = c.execute("SELECT count(*) FROM track_features WHERE COALESCE(status, 'ok') = 'ok'").fetchone()[0]
            conn.close()
        except Exception:
            pass

    total = len(engine.tracks)
    pct = round((analyzed_count / total * 100.0), 1) if total > 0 else 0.0
    return jsonify({
        "analyzed_tracks": analyzed_count,
        "total_tracks": total,
        "completion_percentage": pct
    })


@app.route("/api/reload", methods=["POST"])
def reload_db():
    try:
        acquired = engine.load_database(blocking=False)
        if not acquired:
            return jsonify({"status": "reload already in progress"}), 409
        return jsonify({"status": "reloaded", "total_tracks": len(engine.tracks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    def run_server(port):
        print(f"[Server] Running on {config.HOST}:{port}")
        app.run(host=config.HOST, port=port, debug=False)

    secondary_port = config.SECONDARY_PORT
    if secondary_port and secondary_port != config.PORT:
        t = threading.Thread(target=run_server, args=(secondary_port,), daemon=True)
        t.start()

    run_server(config.PORT)
