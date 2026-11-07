import os
import sys
import time
import threading
import sqlite3
import subprocess
import logging
from flask import Flask, request, jsonify
import config
from engine import AutoplayEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("navidrome_autoplay.server")

start_time = time.time()
_sync_started = False
_sync_lock = threading.Lock()

# Global module-level engine initialized lazily to avoid import-time side effects
engine = AutoplayEngine(lazy=True)


def check_pending_and_prune_features(nav_db_path: str, features_path: str) -> set:
    """Prune orphaned item_ids from track_features and return pending unanalyzed track IDs."""
    import analyzer

    conn = config.connect_ro(nav_db_path)
    try:
        c = conn.cursor()
        nav_ids = [r[0] for r in c.execute("SELECT id FROM media_file").fetchall()]
    finally:
        conn.close()

    nav_id_set = set(nav_ids)
    if os.path.exists(features_path):
        fconn = sqlite3.connect(features_path, timeout=10)
        try:
            fconn.execute("PRAGMA journal_mode=WAL;")
            fc = fconn.cursor()
            tables = {r[0] for r in fc.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "track_features" not in tables:
                return nav_id_set
            feat_ids = [r[0] for r in fc.execute("SELECT item_id FROM track_features").fetchall()]
            orphaned = [fid for fid in feat_ids if fid not in nav_id_set]
            if orphaned:
                fc.executemany("DELETE FROM track_features WHERE item_id = ?", [(oid,) for oid in orphaned])
                fconn.commit()
                logger.info(f"Pruned {len(orphaned)} orphaned tracks from track_features.")

            return analyzer.pending_track_ids(nav_id_set, fconn)
        finally:
            fconn.close()
    return nav_id_set


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
                logger.info(f"Synced {count} analyzed track features into memory.")
        except Exception as e:
            logger.exception(f"Feature sync error: {e}")

        # 2. Check for newly uploaded music every 15 minutes
        if (now - last_analysis_check) > 900:
            last_analysis_check = now
            try:
                if not engine.db_path:
                    continue
                pending = check_pending_and_prune_features(engine.db_path, config.FEATURES_DB_PATH)

                if pending:
                    logger.info(f"Detected {len(pending)} unanalyzed tracks in Navidrome. Spawning background analyzer...")
                    subprocess.Popen(
                        [sys.executable, os.path.join(config.APP_DIR, "analyzer.py"), "2"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except Exception as e:
                logger.exception(f"Background check error: {e}")

        # 3. Reload full database every 2 hours to incorporate fresh scrobbles and ratings
        if (now - last_reload_check) > 7200:
            last_reload_check = now
            try:
                logger.info("Periodic refresh of listening history and graph...")
                engine.load_database(blocking=False)
            except Exception as e:
                logger.exception(f"Periodic reload error: {e}")


def ensure_background_sync(flask_app):
    global _sync_started
    if _sync_started or flask_app.config.get("TESTING"):
        return
    with _sync_lock:
        if not _sync_started:
            sync_thread = threading.Thread(target=auto_feature_sync, daemon=True)
            sync_thread.start()
            _sync_started = True
            logger.info("Background auto_feature_sync thread started.")


def create_app(engine_instance=None, start_sync: bool = False) -> Flask:
    global engine
    if engine_instance is not None:
        engine = engine_instance

    flask_app = Flask(__name__)

    @flask_app.before_request
    def before_request_hook():
        ensure_background_sync(flask_app)
        if not flask_app.config.get("TESTING") and not engine.tracks:
            try:
                if not engine.db_path:
                    engine.db_path = config.resolve_db_path()
                if not engine.tracks:
                    engine.load_database()
            except Exception as e:
                logger.warning(f"Database auto-load on request failed: {e}")

    @flask_app.route("/", methods=["GET"])
    @flask_app.route("/health", methods=["GET"])
    @flask_app.route("/api/status", methods=["GET"])
    def health():
        return jsonify({
            "status": "online",
            "service": "navidrome-autoplay",
            "version": "2.0.0",
            "uptime_seconds": round(time.time() - start_time, 1),
            "total_tracks": len(engine.tracks),
            "total_artists": len(engine.artist_tracks),
            "total_artist_connections": engine.artist_connection_count(),
        })

    @flask_app.route("/api/similar_tracks", methods=["GET"])
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
        return jsonify(results)

    @flask_app.route("/api/similar_artists", methods=["GET"])
    def similar_artists():
        artist_query = request.args.get("artist")
        if not artist_query:
            return jsonify({"error": "artist parameter is required"}), 400

        count = request.args.get("count", default=request.args.get("artistSimilarCount", default=10, type=int), type=int)
        count = max(1, min(count, config.MAX_RESULTS))
        results = engine.get_similar_artists(artist_identifier=artist_query, count=count)
        return jsonify(results)

    @flask_app.route("/api/find_path", methods=["GET"])
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

    @flask_app.route("/api/analysis_status", methods=["GET"])
    def analysis_status():
        analyzed_count = 0
        if os.path.exists(config.FEATURES_DB_PATH):
            try:
                conn = sqlite3.connect(config.FEATURES_DB_PATH, timeout=10)
                try:
                    c = conn.cursor()
                    analyzed_count = c.execute(
                        "SELECT count(*) FROM track_features WHERE COALESCE(status, 'ok') = 'ok'"
                    ).fetchone()[0]
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Error checking analysis status: {e}")

        total = len(engine.tracks)
        pct = round((analyzed_count / total * 100.0), 1) if total > 0 else 0.0
        return jsonify({
            "analyzed_tracks": analyzed_count,
            "total_tracks": total,
            "completion_percentage": pct
        })

    @flask_app.route("/api/reload", methods=["POST"])
    def reload_db():
        # Security: loopback or API key verification
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        remote_addr = request.remote_addr or ""
        is_loopback = remote_addr in ("127.0.0.1", "::1", "localhost", "testclient")
        auth_key = config.AUTOPLAY_API_KEY

        if auth_key:
            if api_key != auth_key:
                return jsonify({"error": "Unauthorized"}), 401
        elif not is_loopback:
            return jsonify({"error": "Forbidden: reload restricted to loopback"}), 403

        try:
            acquired = engine.load_database(blocking=False)
            if not acquired:
                return jsonify({"status": "reload already in progress"}), 409
            return jsonify({"status": "reloaded", "total_tracks": len(engine.tracks)})
        except Exception as e:
            logger.exception(f"Reload database error: {e}")
            return jsonify({"error": "Internal server error"}), 500

    if start_sync:
        ensure_background_sync(flask_app)

    return flask_app


# Backward-compatible module-level app
app = create_app(engine_instance=engine, start_sync=False)


def run_production_server():
    bind_addrs = [f"{config.HOST}:{config.PORT}"]
    if config.SECONDARY_PORT and config.SECONDARY_PORT != config.PORT:
        bind_addrs.append(f"{config.HOST}:{config.SECONDARY_PORT}")

    ensure_background_sync(app)

    try:
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, application, options=None):
                self.options = options or {}
                self.application = application
                super().__init__()

            def load_config(self):
                cfg = {key: value for key, value in self.options.items()
                       if key in self.cfg.settings and value is not None}
                for key, value in cfg.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        logger.info(f"Starting Gunicorn on {', '.join(bind_addrs)}")
        opts = {
            "bind": bind_addrs,
            "workers": 1,
            "threads": 4,
            "worker_class": "gthread",
            "timeout": 30,
            "accesslog": "-",
            "errorlog": "-",
        }
        StandaloneApplication(app, opts).run()
    except ImportError:
        try:
            from waitress import serve
            logger.info(f"Starting Waitress on {config.HOST}:{config.PORT}")
            serve(app, host=config.HOST, port=config.PORT)
        except ImportError:
            logger.warning("Neither Gunicorn nor Waitress installed, falling back to Flask dev server.")
            secondary_port = config.SECONDARY_PORT
            if secondary_port and secondary_port != config.PORT:
                t = threading.Thread(
                    target=lambda: app.run(host=config.HOST, port=secondary_port, debug=False),
                    daemon=True
                )
                t.start()
            app.run(host=config.HOST, port=config.PORT, debug=False)


if __name__ == "__main__":
    run_production_server()
