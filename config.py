import os

# Database configuration
PRIMARY_DB_PATH = os.environ.get("NAVIDROME_DB", "/var/lib/navidrome/navidrome.db")
BACKUP_DB_PATH = "/root/navidrome_backup_20261106_194244/navidrome.db"

# Server configuration
HOST = os.environ.get("AUTOPLAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("AUTOPLAY_PORT", "8001"))
DEBUG = os.environ.get("AUTOPLAY_DEBUG", "false").lower() == "true"

# Recommendation & Pacing weights
MACRO_ANCHOR_WEIGHT = float(os.environ.get("MACRO_ANCHOR_WEIGHT", "0.30"))
LOCAL_TRANSITION_WEIGHT = float(os.environ.get("LOCAL_TRANSITION_WEIGHT", "0.45"))
ARTIST_AFFINITY_WEIGHT = float(os.environ.get("ARTIST_AFFINITY_WEIGHT", "0.25"))

# Stochastic Exploration / Serendipity (Spotify BaRT style)
# Controls shuffle entropy so repeated Instant Mixes on the same track rotate through diverse viable picks
RANDOMIZATION_TEMPERATURE = float(os.environ.get("AUTOPLAY_TEMPERATURE", "0.20"))

# Fatigue & Pacing rules
MAX_ARTIST_REPEAT_IN_QUEUE = int(os.environ.get("MAX_ARTIST_REPEAT_IN_QUEUE", "2"))
ARTIST_FATIGUE_DECAY = float(os.environ.get("ARTIST_FATIGUE_DECAY", "0.60"))
BPM_TOLERANCE_PCT = float(os.environ.get("BPM_TOLERANCE_PCT", "0.10"))  # 10% tempo transition window
FAMILIAR_EXPLOITATION_RATIO = float(os.environ.get("FAMILIAR_RATIO", "0.75"))  # 75% familiar, 25% discovery

# In-memory session tracking TTL (seconds)
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "7200"))
