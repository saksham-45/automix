"""
Centralized runtime settings.
Values come from environment variables with safe defaults for production.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; fall back to default on bad input so a malformed
    value doesn't crash every module that imports settings at import time."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


# Core mix defaults
TRANSITION_LENGTH = _env_float("TRANSITION_LENGTH", 30)
TRANSITION_WINDOW = _env_float("TRANSITION_WINDOW", 64)
LUFS_TARGET = _env_float("LUFS_TARGET", -14)
SR = _env_int("SAMPLE_RATE", 44100)

# Feature toggles
ENABLE_AGENTIC_DURATION = os.getenv("ENABLE_AGENTIC_DURATION", "false").lower() in ("1", "true", "yes")

# Paths
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "data" / "cache" / "stream"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / "temp_audio"))

# Server
# PORT defaults to 7860 (Hugging Face Spaces' expected container port); override
# locally with PORT=5005. HOST binds all interfaces so it works inside a container.
PORT = _env_int("PORT", 7860)
HOST = os.getenv("HOST", "0.0.0.0")

# YouTube / network
YTDLP_EXTRACTOR_ARGS = os.getenv("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=android")
DOWNLOAD_TIMEOUT_SEC = _env_int("DOWNLOAD_TIMEOUT_SEC", 120)
# Optional Netscape-format cookies file for yt-dlp. Datacenter IPs (any cloud host)
# are bot-throttled by YouTube; supplying your own cookies via this path is the
# standard mitigation. On Hugging Face Spaces, store the file contents in a Secret
# and write them to this path at startup. Empty => no cookies (samples still work).
YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip()

# --- Abuse / resource caps (memory-efficiency + security) ------------------- #
# Hard limits so a single request can't exhaust RAM, disk, CPU, or download
# bandwidth on a small/free cloud box. All overridable via env.
ALLOW_YOUTUBE = os.getenv("ALLOW_YOUTUBE", "true").lower() in ("1", "true", "yes")
MAX_PLAYLIST_TRACKS = _env_int("MAX_PLAYLIST_TRACKS", 12)   # cap tracks per set
# Seconds fetched per track. 300 lets a song reach its outro (where a DJ mixes out);
# the hosted demo image lowers it to 150 to stay within a free box (see Dockerfile).
PER_TRACK_CAP_SEC = _env_int("PER_TRACK_CAP_SEC", 300)
MAX_DOWNLOAD_MB = _env_int("MAX_DOWNLOAD_MB", 60)           # yt-dlp --max-filesize
SESSION_TTL_SEC = _env_int("SESSION_TTL_SEC", 1800)        # evict idle sessions after 30 min
MAX_SESSIONS = _env_int("MAX_SESSIONS", 24)                # cap concurrent sessions
RATE_LIMIT = os.getenv("RATE_LIMIT", "30 per hour")        # per-IP on the start endpoint


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Render queue ---------------------------------------------------------- #
# Renders are CPU-heavy (and demucs is heavier still), so only this many run at
# once; the rest wait in a bounded line and get a queue position. Raise on a GPU.
MAX_CONCURRENT_RENDERS = max(1, _env_int("MAX_CONCURRENT_RENDERS", 1))
MAX_QUEUE_LENGTH = max(0, _env_int("MAX_QUEUE_LENGTH", 4))   # waiting jobs; beyond -> 503

# --- Uploads (bring your own audio) ----------------------------------------- #
ALLOW_UPLOADS = _env_bool("ALLOW_UPLOADS", True)
MAX_UPLOAD_MB = max(1, _env_int("MAX_UPLOAD_MB", 60))       # per file
UPLOAD_TTL_SEC = _env_int("UPLOAD_TTL_SEC", 3600)           # delete unused uploads after 1 h
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "120 per hour")  # per-IP on /api/upload
MIN_TRACK_SEC = _env_float("MIN_TRACK_SEC", 30)              # shorter tracks are skipped

# --- Disk / subprocess bounds ---------------------------------------------- #
MAX_CACHE_GB = _env_float("MAX_CACHE_GB", 5)                 # CACHE_DIR cap; oldest tracks evicted
FFMPEG_TIMEOUT_SEC = _env_int("FFMPEG_TIMEOUT_SEC", 120)     # every ffmpeg/ffprobe call

# --- Observability / HTTP ---------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON = _env_bool("LOG_JSON", False)                       # one JSON object per log line
# Number of reverse proxies in front of the app (HF Spaces / Fly / nginx = 1). Needed
# so per-IP rate limits see the client IP instead of the proxy's.
TRUST_PROXY_HOPS = max(0, _env_int("TRUST_PROXY_HOPS", 0))
# CSP frame-ancestors. 'none' forbids framing; HF Spaces renders the app inside an
# iframe on huggingface.co, so set "https://huggingface.co" there.
def _csp_sources(raw: str) -> str:
    # Accept none/self with or without the CSP quotes (dotenv strips quotes).
    out = []
    for tok in raw.split():
        bare = tok.strip("'\"").lower()
        out.append(f"'{bare}'" if bare in ("none", "self") else tok)
    return " ".join(out) or "'none'"


FRAME_ANCESTORS = _csp_sources(os.getenv("FRAME_ANCESTORS", "'none'"))
