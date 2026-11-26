import os
import math

# Base directory of the application (used for runtime files such as features.db)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Database configuration
PRIMARY_DB_PATH = os.environ.get("NAVIDROME_DB", "/var/lib/navidrome/navidrome.db")
# Optional fallback copy of the Navidrome DB (e.g. a backup snapshot). Empty = disabled.
BACKUP_DB_PATH = os.environ.get("NAVIDROME_DB_BACKUP", "")
FEATURES_DB_PATH = os.environ.get("AUTOPLAY_FEATURES_DB", os.path.join(APP_DIR, "features.db"))
MUSIC_DIR = os.environ.get("NAVIDROME_MUSIC_DIR", "/opt/navidrome/music")

# Server configuration
# This service has no Web UI and only serves JSON to the local Navidrome plugin.
# Binding to 127.0.0.1 (localhost) is sufficient and recommended when running on
# the same machine as Navidrome. Only set AUTOPLAY_HOST=0.0.0.0 if Navidrome runs
# in a separate container/VM without host network access.
HOST = os.environ.get("AUTOPLAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("AUTOPLAY_PORT", "8001"))
# Optional second port for compatibility with plugin configs pointing at 8000. 0 = disabled.
SECONDARY_PORT = int(os.environ.get("AUTOPLAY_SECONDARY_PORT", "8000"))

# Request limits (protect the service from accidental / malicious heavy requests)
MAX_RESULTS = max(1, int(os.environ.get("AUTOPLAY_MAX_RESULTS", "200")))
MAX_PATH_STEPS = max(2, int(os.environ.get("AUTOPLAY_MAX_PATH_STEPS", "25")))

# Recommendation & Pacing weights
MACRO_ANCHOR_WEIGHT = float(os.environ.get("MACRO_ANCHOR_WEIGHT", "0.40"))
LOCAL_TRANSITION_WEIGHT = float(os.environ.get("LOCAL_TRANSITION_WEIGHT", "0.40"))

# Stochastic Exploration / Serendipity (Spotify BaRT style)
# Controls shuffle entropy so repeated Instant Mixes on the same track rotate through diverse viable picks
_raw_temp = float(os.environ.get("AUTOPLAY_TEMPERATURE", "0.10"))
RANDOMIZATION_TEMPERATURE = _raw_temp if (math.isfinite(_raw_temp) and _raw_temp >= 0.0) else 0.10

# Fatigue & Pacing rules
MAX_ARTIST_REPEAT_IN_QUEUE = max(1, int(os.environ.get("MAX_ARTIST_REPEAT_IN_QUEUE", "2")))
BPM_TOLERANCE_PCT = float(os.environ.get("BPM_TOLERANCE_PCT", "0.10"))  # 10% tempo transition window

# Multi-user privacy: optionally restrict playlist graph edges to public playlists only
PUBLIC_PLAYLISTS_ONLY = os.environ.get("AUTOPLAY_PUBLIC_PLAYLISTS_ONLY", "false").lower() == "true"

# In-memory session tracking TTL (seconds) and LRU capacity cap
SESSION_TTL_SECONDS = max(1, int(os.environ.get("SESSION_TTL_SECONDS", "7200")))
MAX_SESSIONS = max(1, int(os.environ.get("MAX_SESSIONS", "1000")))
MAX_SESSION_ID_LEN = max(16, int(os.environ.get("MAX_SESSION_ID_LEN", "128")))
AUTOPLAY_API_KEY = os.environ.get("AUTOPLAY_API_KEY", "")


def resolve_db_path() -> str:
    """Single source of truth for which Navidrome DB to read (used by engine AND analyzer)."""
    if os.path.exists(PRIMARY_DB_PATH) and os.access(PRIMARY_DB_PATH, os.R_OK):
        return PRIMARY_DB_PATH
    if BACKUP_DB_PATH and os.path.exists(BACKUP_DB_PATH):
        return BACKUP_DB_PATH
    raise FileNotFoundError(
        f"Navidrome database not found at {PRIMARY_DB_PATH}"
        + (f" or {BACKUP_DB_PATH}" if BACKUP_DB_PATH else "")
    )


def connect_ro(path: str):
    """Open an SQLite database read-only (never modify Navidrome's DB)."""
    import sqlite3
    from urllib.parse import quote
    if not path or not os.path.isfile(path):
        raise sqlite3.OperationalError(f"unable to open database file: {path}")
    quoted = quote(os.path.abspath(path), safe="/")
    return sqlite3.connect(f"file:{quoted}?mode=ro", uri=True, timeout=10)
