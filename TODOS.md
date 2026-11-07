# Hardening & Grading Report (`TODOS.md`)

**Overall Grade: A**

All architectural, security, concurrency, algorithmic, and testing hardening tasks have been resolved.

| Category | Grade | Summary |
| :--- | :---: | :--- |
| **Algorithm & Data Modeling** | **A** | Native Navidrome genres loaded into startup state; token/delimiter matching prevents substring false positives; 0.5000 RMS energy normalization fixed; graph edges and SVD embeddings rebuilt incrementally on feature reloads. |
| **Concurrency & State** | **A** | Server import-time side effects removed with app factory and lazy initialization; bounded OrderedDict LRU session cache (capped at `MAX_SESSIONS`); refined session reset heuristic preserving seed anchor for manual queueing; SQLite WAL mode enabled on features.db with try/finally connection closures; production WSGI server (Gunicorn) support. |
| **Security & Operations** | **A** | Service runs as unprivileged `navidrome` user with full systemd sandboxing (`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`, `ProtectHome=true`, `ReadWritePaths`); path traversal prevention in `analyzer.py`; no filesystem path leaks in status/health endpoints; `/api/reload` loopback restriction and `AUTOPLAY_API_KEY` support with generic 500 responses; hardened `install.sh` and `service.sh`. |
| **Reliability & Edge Cases** | **A** | Unanalyzed track detection uses `analyzer.pending_track_ids()` and prunes orphaned IDs; empty (0 tracks) and single-track (1 track) libraries guarded across SVD embeddings, beam search, and similarity recommendations. |
| **Testing & Maintainability** | **A** | 27 unit and integration tests covering audio analysis, path traversal, lock acquisition, orphan pruning, session expiry & caps, edge cases, reload authorization, and error handling; structured `logging` across all components. |

---

## 1. Bugs & Correctness Issues (High Priority)

- [x] **Load native Navidrome genres in `build_state()` (`engine.py:323-349`)**
  - `build_state()` queries `SELECT id, title, artist, album, artist_id, album_id, year, bpm, duration, path FROM media_file`, omitting `genre` (and/or `media_file_genres`), and hardcodes `'genre': ''` on line 343.
  - Unless a track has already been processed by `analyzer.py` into `features.db`, `get_macro_cluster()` has zero genre information and falls back to `'pop'` for almost the entire library on initial startup.

- [x] **Fix unanalyzed track detection in `auto_feature_sync()` (`server.py:41-64`)**
  - Currently compares `SELECT count(*) FROM media_file` against `SELECT count(*) FROM track_features`.
  - If tracks are deleted from Navidrome, stale rows remain in `track_features`, inflating `total_in_features` so newly added tracks are missed (`total_in_navidrome > total_in_features` stays `False`).
  - Conversely, if counts diverge or tracks fail before insertion, `server.py` spawns `analyzer.py` every 15 minutes with `stderr=DEVNULL`.
  - **Fix:** Reuse `analyzer.pending_track_ids()` (comparing actual track ID sets and retry cutoffs) and prune orphaned `item_id`s from `track_features`.

- [x] **Handle graph/embedding staleness in `reload_features()` (`engine.py:606-626`)**
  - `reload_features()` updates `st.tracks` and recomputes feature arrays (`a_bpm`, `a_cluster`, `a_energy`, `a_cam_*`, `a_genre`), but does not rebuild the `5c. Cluster BPM neighbours` graph edges or recompute SVD `st.embeddings` until the 2-hour full reload.

- [x] **Fix single-track and `0.5` magic-number energy normalization (`engine.py:279-287`)**
  - Line 279 (`abs(fen - 0.5) > 1e-9`) drops any track whose legitimate RMS energy happens to be `0.5000`. Since `status` now tracks decode errors, rely on `status == 'ok'` instead of magic float comparisons.
  - When `len(vals) == 1`, `ranks = vals.argsort().argsort() / max(1, len(vals) - 1)` assigns percentile `0.0` instead of `0.5`.

- [x] **Replace substring genre matching with token/delimiter matching (`engine.py:706-708`)**
  - `gmatch = np.array([g0 == g or g0 in g or g in g0 for g in st.genres] + [False])` uses raw substring checks (`g0 in g or g in g0`), causing `"pop"` to match `"j-pop"`, `"k-pop"`, `"synthpop"`, and `"austropop"`, and `"rock"` to match `"country rock"` or `"punk rock"`.

- [x] **Guard empty and 1-track library edge cases (`engine.py:542-547`, `engine.py:885`)**
  - In `find_path()`, `np.argpartition(-scores, min(12, len(scores) - 1))` raises `ValueError` when `len(scores) == 0`, or selects the `-inf` current track when `len(scores) == 1`.
  - In `build_state()`, fallback `st.embeddings` has shape `(n, 16)` when `n_components <= 2`, whereas normal SVD output has shape `(n, min(32, n - 1))`.

---

## 2. Concurrency, Lifecycle & Resource Management

- [x] **Remove import-time side effects in `server.py` (`server.py:14-16`, `server.py:76-77`)**
  - Importing `server` immediately instantiates `AutoplayEngine()` (reading `/var/lib/navidrome/navidrome.db`) and starts the `auto_feature_sync` daemon thread before tests or WSGI runners can configure the app.
  - Wrap initialization and background thread startup in an app factory (`create_app()`) or lazy initialization guard.

- [x] **Cap `self.sessions` with a bounded LRU cache (`engine.py:743-759`)**
  - `self.sessions` only evicts by `SESSION_TTL_SECONDS` (7200s) and has no max size limit. Because `key = session_id or item_id`, high request volume with distinct IDs can grow `self.sessions` unboundedly in memory, and every request scans all sessions under `self._sess_lock`.
  - Use `collections.OrderedDict` with a `MAX_SESSIONS` cap (e.g., 1,000) in addition to TTL eviction.

- [x] **Refine session reset heuristic (`engine.py:752`)**
  - `if sess is None or (item_id not in sess.history and item_id not in sess.recent_recs):` wipes `seed_id` and `history` whenever a user manually inserts a track into their queue during an active `session_id`. Distinguish between explicit `session_id` tracking and implicit `item_id`-keyed calls.

- [x] **Enable SQLite WAL mode on `features.db` and use context managers (`analyzer.py:40-60`, `server.py:41-52`, `server.py:155-160`)**
  - Enable `PRAGMA journal_mode=WAL;` in `init_features_db()` so `server.py` reads every 15s don't contend with `analyzer.py` batch writes.
  - Ensure all SQLite connections in `server.py` and `analyzer.py` are closed inside `try / finally` blocks.

- [x] **Use a production WSGI server (`server.py:183-193`, `requirements.txt`)**
  - Replace dual-thread `app.run()` (Werkzeug development server) with `gunicorn` or `waitress` for production durability, request timeouts, and clean `SIGTERM` handling.

---

## 3. Security & Operational Hardening

- [x] **Run service as an unprivileged user and harden systemd unit (`navidrome-autoplay.service`, `install.sh:60-79`)**
  - Create/use a dedicated service user (or `navidrome` user) instead of `User=root`.
  - Add systemd sandboxing directives (`NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectSystem=strict`, `ProtectHome=true`, `ReadWritePaths=...`).

- [x] **Prevent path traversal in `analyzer.py` (`analyzer.py:174`)**
  - `os.path.join(MUSIC_DIR, rel_path or '')` allows absolute `rel_path` values or `..` segments to escape `MUSIC_DIR`. Normalize via `os.path.realpath` and verify `os.path.commonpath([real_music_dir, real_full_path]) == real_music_dir`.

- [x] **Stop leaking filesystem paths and internal exception strings (`server.py:92`, `server.py:179-180`)**
  - Remove `"database_path": engine.db_path` from `/health` and `/api/status` responses.
  - Log full traceback server-side in `/api/reload` and return a generic `500` error message to clients.
  - Optional: Support an `AUTOPLAY_API_KEY` or restrict `/api/reload` to loopback (`127.0.0.1` / `::1`) even when `AUTOPLAY_HOST=0.0.0.0`.

- [x] **Harden `install.sh` (`install.sh:34-52`)**
  - Use a local virtual environment (`$APP_DIR/.venv`) instead of `pip3 install --break-system-packages`.
  - Do not swallow package manager errors with `|| true`.
  - Pass `$APP_DIR` via environment variable instead of interpolating it directly into `python3 -c "..."`.

- [x] **Verify process identity before killing in `service.sh` (`service.sh:35-56`)**
  - Before sending `SIGTERM` / `SIGKILL` to the PID in `autoplay.pid`, verify `/proc/$PID/cmdline` contains `server.py` to avoid killing recycled PIDs.

---

## 4. Testing & Code Hygiene

- [x] **Expand unit and integration test coverage (`test_autoplay.py`, `test_server.py`)**
  - Test `analyzer.py`: `extract_key_from_pcm()` (synthetic sine/chord wave), `analyze_track()` (missing file, corrupt file, valid audio), `pending_track_ids()` (24h retry backoff), and `acquire_single_instance_lock()`.
  - Test `engine.reload_features()` incremental updates.
  - Test session TTL expiration, max session cap, `MAX_ARTIST_REPEAT_IN_QUEUE`, and `eliminate_duplicates=True/False`.
  - Test empty database (`0` tracks) and single-track (`1` track) edge cases across `/api/similar_tracks` and `/api/find_path`.

- [x] **Replace `print()` with structured `logging` (`engine.py`, `server.py`, `analyzer.py`)**
  - Configure standard Python `logging` with timestamps, levels (`INFO`, `WARNING`, `ERROR`), and `logger.exception()` where errors are caught.
