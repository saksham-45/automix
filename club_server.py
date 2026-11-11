#!/usr/bin/env python3
"""
Club AutoMixer server — paste a YouTube playlist, upload your own tracks, or use the
bundled sample, and get ONE continuous, gapless, Boiler-Room-style mixed set that
starts playing while it is still rendering.

Run locally (use the project venv with matched torch/torchaudio + librosa + demucs):
    PORT=5005 ./.venv/bin/python club_server.py
Then open http://localhost:5005

In a container it is served by gunicorn with a SINGLE worker + threads (see
gunicorn.conf.py). Single worker because session state, the render queue, the
upload registry and the chunk index all live in this process's memory.

Streaming model: the mixed timeline is sample-contiguous by construction, so the
producer emits it as fixed-size PCM chunks AS IT RENDERS (via build_continuous_set's
on_part callback). The web client schedules chunks back-to-back with the Web Audio
API and starts playback once a few chunks are buffered.

Bounded work: renders run on a fixed pool (MAX_CONCURRENT_RENDERS) behind a bounded
wait line (MAX_QUEUE_LENGTH, beyond which /api/set/start answers 503). A render can
be cancelled (DELETE /api/set/<sid>): the on_part / progress / loader callbacks raise
a private BaseException, build_continuous_set unwinds, the slot frees and the chunk
files are deleted. Abandoned renders (no client poll for ABANDON_AFTER_SEC) are
cancelled by the janitor.

Security: only YouTube URLs are accepted (SSRF guard); video ids are whitelisted to
the canonical 11-char form before use in filenames (path-traversal guard); uploads
are content-sniffed with ffprobe against an allow-list of audio containers, stored
under random server-side names and decoded with ffmpeg restricted to the file
protocol; every subprocess has a timeout; error text sent to clients never contains
exception strings or filesystem paths; strict CSP + security headers on every response.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from flask import Flask, g, jsonify, request, send_file, render_template
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.club_mixer import build_continuous_set, _as_stereo, stems_available  # noqa: E402
from src.demo_samples import ensure_demo_samples  # noqa: E402
from src.jobs import Cancelled, JobQueue, QueueFull  # noqa: E402
from src.settings import (  # noqa: E402
    SR, CACHE_DIR, HOST, PORT, DOWNLOAD_TIMEOUT_SEC, YTDLP_EXTRACTOR_ARGS,
    YT_COOKIES_FILE, ALLOW_YOUTUBE, MAX_PLAYLIST_TRACKS, PER_TRACK_CAP_SEC,
    MAX_DOWNLOAD_MB, SESSION_TTL_SEC, MAX_SESSIONS, RATE_LIMIT,
    MAX_CONCURRENT_RENDERS, MAX_QUEUE_LENGTH, ALLOW_UPLOADS, MAX_UPLOAD_MB,
    UPLOAD_TTL_SEC, UPLOAD_RATE_LIMIT, MIN_TRACK_SEC, MAX_CACHE_GB,
    FFMPEG_TIMEOUT_SEC, LOG_LEVEL, LOG_JSON, TRUST_PROXY_HOPS, FRAME_ANCESTORS,
)

ABANDON_AFTER_SEC = int(os.getenv("ABANDON_AFTER_SEC", "120"))
JANITOR_INTERVAL_SEC = 30
CHUNK_SEC = 4.0
MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# logging — every line carries the session id (rid) it belongs to
# --------------------------------------------------------------------------- #
_rid: contextvars.ContextVar = contextvars.ContextVar("rid", default="-")


class _RidFilter(logging.Filter):
    def filter(self, record):
        record.rid = _rid.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
               "logger": record.name, "rid": getattr(record, "rid", "-"),
               "msg": record.getMessage()}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def _setup_logging() -> logging.Logger:
    base = logging.getLogger("automix")
    if not getattr(base, "_automix_configured", False):
        h = logging.StreamHandler(sys.stderr)
        h.addFilter(_RidFilter())
        h.setFormatter(_JsonFormatter() if LOG_JSON else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s rid=%(rid)s %(message)s"))
        base.addHandler(h)
        base.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        base.propagate = False
        base._automix_configured = True
    return logging.getLogger("automix.server")


log = _setup_logging()

# --------------------------------------------------------------------------- #
# app + directories
# --------------------------------------------------------------------------- #
app = Flask(__name__, template_folder=str(ROOT / "templates"))
# One request carries at most MAX_PLAYLIST_TRACKS files of MAX_UPLOAD_MB each.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * MB * MAX_PLAYLIST_TRACKS + MB
app.json.sort_keys = False

if TRUST_PROXY_HOPS:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUST_PROXY_HOPS, x_proto=TRUST_PROXY_HOPS)

CHUNK_DIR = CACHE_DIR / "chunks"
DL_DIR = CACHE_DIR / "tracks"
UPLOAD_DIR = CACHE_DIR / "uploads"
DEMO_DIR = ROOT / "data" / "demo"
for d in (CHUNK_DIR, DL_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Optional per-IP rate limiting on the expensive endpoints. No-op if flask-limiter
# isn't installed, so the server still runs in a minimal environment.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")
    _rate_limit = limiter.limit(RATE_LIMIT)
    _upload_limit = limiter.limit(UPLOAD_RATE_LIMIT)
except Exception:  # flask-limiter missing or failed to init
    limiter = None

    def _rate_limit(fn):
        return fn
    _upload_limit = _rate_limit

# In-process state (hence the single gunicorn worker).
SESSIONS: dict = {}
UPLOADS: dict = {}
_SLOCK = threading.RLock()
_IN_USE: dict = {}                 # track path -> refcount (protects from eviction)
_DL_LOCKS: dict = {}               # video id -> lock (no duplicate concurrent downloads)
JOBS = JobQueue(MAX_CONCURRENT_RENDERS, MAX_QUEUE_LENGTH)

_SID_RE = re.compile(r"[0-9a-f]{32}")
_UPLOAD_ID_RE = re.compile(r"[0-9a-f]{32}")
_VIDEO_ID_RE = re.compile(r"[0-9A-Za-z_-]{11}")
_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
                  "music.youtube.com", "youtu.be", "www.youtu.be"}
# ffprobe format_name tokens we accept. Anything else (playlists like hls/concat that
# make ffmpeg open other files or URLs, images, text "tty", ...) is rejected.
_AUDIO_FORMATS = {"mp3", "wav", "w64", "flac", "ogg", "mov", "mp4", "m4a", "aiff",
                  "matroska", "webm", "aac", "caf", "wv", "ape", "asf"}
_SAMPLE_SOURCES = ["data/demo/sample_a.wav", "data/demo/sample_b.wav", "data/demo/sample_c.wav"]


class UserError(Exception):
    """An error whose message is safe to show to the client (no paths, no internals)."""


class TrackError(UserError):
    """One track could not be used; the set continues without it."""


# --------------------------------------------------------------------------- #
# validation / security helpers
# --------------------------------------------------------------------------- #
def _is_youtube_url(url: str) -> bool:
    """SSRF guard: only accept http(s) URLs whose host is a known YouTube domain,
    so the download subprocess can never be pointed at an internal/arbitrary host."""
    try:
        u = urlparse(url)
    except Exception:
        return False
    return u.scheme in ("http", "https") and (u.hostname or "").lower() in _ALLOWED_HOSTS


def _safe_vid(raw: str) -> str:
    """Extract the canonical 11-char YouTube id from a url/id and reject anything
    else. Used to build cache filenames -> path-traversal / injection guard."""
    tail = raw.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    m = _VIDEO_ID_RE.fullmatch(tail) or _VIDEO_ID_RE.search(raw)
    if not m:
        raise TrackError("not a valid YouTube video id")
    return m.group(0)


def _clean_name(name) -> str:
    """Display-only version of a client filename (never used on disk)."""
    name = Path(str(name or "upload")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip() or "upload"
    return name[:120]


def _which(binary: str):
    return shutil.which(binary)


def _binaries() -> dict:
    out = {"ffmpeg": bool(_which("ffmpeg")), "ffprobe": bool(_which("ffprobe"))}
    if ALLOW_YOUTUBE:
        out["yt-dlp"] = bool(_which("yt-dlp"))
    return out


def _run(cmd: list, timeout: float, what: str) -> subprocess.CompletedProcess:
    """subprocess.run with a mandatory timeout; raises TrackError with a safe message."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise TrackError(f"{what} timed out")
    except FileNotFoundError:
        log.error("%s is not installed (needed for %s)", cmd[0], what)
        raise TrackError(f"{what} is unavailable on this server")


# --------------------------------------------------------------------------- #
# audio ingest: probe + decode (shared by uploads and downloads)
# --------------------------------------------------------------------------- #
def _probe_audio(path: Path) -> None:
    """Content-sniff with ffprobe (the extension is never trusted)."""
    r = _run(["ffprobe", "-v", "error", "-protocol_whitelist", "file",
              "-print_format", "json",
              "-show_entries", "format=format_name:stream=codec_type",
              "-i", f"file:{path}"], FFMPEG_TIMEOUT_SEC, "probing the file")
    try:
        info = json.loads(r.stdout or "{}")
    except ValueError:
        info = {}
    fmt = set((info.get("format") or {}).get("format_name", "").split(","))
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams") or [])
    if r.returncode != 0 or not has_audio or not (fmt & _AUDIO_FORMATS):
        raise TrackError("not a recognised audio file")


def _decode_to_wav(src: Path, dst: Path) -> float:
    """Decode the first audio stream to 16-bit stereo WAV at settings.SR, capped at
    PER_TRACK_CAP_SEC. Written to a temp name then renamed, so a half-written file is
    never mistaken for a cached track. Returns the duration in seconds."""
    tmp = dst.with_name(dst.name + ".part.wav")
    r = _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-protocol_whitelist", "file", "-i", f"file:{src}",
              "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", str(PER_TRACK_CAP_SEC),
              "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", "-f", "wav", f"file:{tmp}"],
             FFMPEG_TIMEOUT_SEC, "decoding")
    try:
        if r.returncode != 0 or not tmp.exists():
            log.warning("ffmpeg decode failed rc=%s: %s", r.returncode, (r.stderr or "")[-300:])
            raise TrackError("could not decode the audio")
        try:
            dur = sf.info(str(tmp)).duration
        except Exception:
            raise TrackError("could not decode the audio")
        os.replace(tmp, dst)
        return float(dur)
    finally:
        tmp.unlink(missing_ok=True)


def _load_wav(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SR:      # bundled samples generated at another rate: resample once
        import librosa
        y = librosa.resample(y.T if y.ndim == 2 else y, orig_sr=sr, target_sr=SR)
        y = y.T if y.ndim == 2 else y
    return y


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
def _ytdlp_base() -> list:
    """Common yt-dlp flags incl. optional cookies (the datacenter-IP bot mitigation)."""
    base = ["yt-dlp", "--no-warnings", "--extractor-args", YTDLP_EXTRACTOR_ARGS]
    if YT_COOKIES_FILE and Path(YT_COOKIES_FILE).is_file():
        base += ["--cookies", YT_COOKIES_FILE]
    return base


def _resolve_playlist(url: str) -> list:
    try:
        r = _run(_ytdlp_base() + ["--flat-playlist", "--get-id", url],
                 DOWNLOAD_TIMEOUT_SEC, "reading the playlist")
    except TrackError as e:
        raise UserError(str(e))
    ids = []
    for v in r.stdout.splitlines():
        v = v.strip()
        if _VIDEO_ID_RE.fullmatch(v) and v not in ids:
            ids.append(v)
    if not ids:
        log.warning("playlist resolve failed rc=%s: %s", r.returncode, (r.stderr or "")[-300:])
        raise UserError("could not read that playlist (private, empty, or blocked by YouTube)")
    return ids[:MAX_PLAYLIST_TRACKS]


def _ytdlp_reason(stderr: str) -> str:
    s = (stderr or "").lower()
    if "confirm you" in s and "bot" in s:
        return "YouTube blocked the download (bot check)"
    if "max-filesize" in s or "larger than max" in s:
        return f"larger than {MAX_DOWNLOAD_MB} MB"
    if "private" in s or "unavailable" in s or "removed" in s:
        return "video unavailable"
    if "age" in s and "sign in" in s:
        return "age-restricted video"
    return "download failed"


def _download(vid: str) -> Path:
    dst = DL_DIR / f"{vid}_{PER_TRACK_CAP_SEC}.wav"     # cap in the name: changing it re-fetches
    with _SLOCK:
        lock = _DL_LOCKS.setdefault(vid, threading.Lock())
    with lock:
        if dst.exists():
            os.utime(dst)                                  # LRU for cache eviction
            return dst
        tmpl = DL_DIR / f"{vid}.src.%(ext)s"
        try:
            r = _run(_ytdlp_base() + [
                "-x", "--audio-format", "wav", "--no-playlist",
                "--max-filesize", f"{MAX_DOWNLOAD_MB}M",
                "--download-sections", f"*0-{PER_TRACK_CAP_SEC}",
                "-o", str(tmpl), f"https://www.youtube.com/watch?v={vid}"],
                DOWNLOAD_TIMEOUT_SEC, "download")
            cand = sorted(p for p in DL_DIR.glob(f"{vid}.src.*") if not p.name.endswith(".part"))
            if r.returncode != 0 or not cand:
                log.warning("yt-dlp failed vid=%s rc=%s: %s", vid, r.returncode,
                            (r.stderr or "")[-300:])
                raise TrackError(_ytdlp_reason(r.stderr))
            _decode_to_wav(cand[0], dst)
            return dst
        finally:
            for p in DL_DIR.glob(f"{vid}.src.*"):
                p.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# disk accounting
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _enforce_cache_cap(extra_bytes: int = 0) -> bool:
    """Evict downloaded tracks (oldest first, never one in use) until CACHE_DIR plus
    `extra_bytes` fits under MAX_CACHE_GB. Returns True if it fits."""
    cap = int(MAX_CACHE_GB * 1024 ** 3)
    used = _dir_size(CACHE_DIR)
    if used + extra_bytes <= cap:
        return True
    files = []
    for p in DL_DIR.glob("*.wav"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            pass
    for _, p in sorted(files):
        with _SLOCK:
            if _IN_USE.get(str(p)):
                continue
        try:
            size = p.stat().st_size
            p.unlink()
            used -= size
            log.info("cache evict %s (%.1f MB)", p.name, size / MB)
        except OSError:
            continue
        if used + extra_bytes <= cap:
            return True
    return used + extra_bytes <= cap


def _acquire(paths) -> None:
    with _SLOCK:
        for p in paths:
            _IN_USE[str(p)] = _IN_USE.get(str(p), 0) + 1


def _release(paths) -> None:
    with _SLOCK:
        for p in paths:
            n = _IN_USE.get(str(p), 0) - 1
            if n <= 0:
                _IN_USE.pop(str(p), None)
            else:
                _IN_USE[str(p)] = n


# --------------------------------------------------------------------------- #
# sessions: discard / reap
# --------------------------------------------------------------------------- #
def _delete_chunk_files(sid: str) -> None:
    for p in CHUNK_DIR.glob(f"{sid}_*.wav"):
        p.unlink(missing_ok=True)


def _discard_session(sid: str, why: str) -> bool:
    """Forget a session: cancel its job (queued -> dropped, running -> unwinds at its
    next callback) and delete its chunk files."""
    with _SLOCK:
        s = SESSIONS.pop(sid, None)
    if s is None:
        return False
    JOBS.cancel(sid)
    _delete_chunk_files(sid)
    token = _rid.set(sid)
    log.info("session discarded (%s) state=%s", why, s.get("state"))
    _rid.reset(token)
    return True


def _delete_upload(uid: str) -> None:
    with _SLOCK:
        u = UPLOADS.pop(uid, None)
    if u:
        Path(u["path"]).unlink(missing_ok=True)


def _reap() -> None:
    """Bound RAM + disk: drop idle/abandoned sessions (and cancel their renders),
    cap the session count, expire unused uploads and enforce the cache size cap."""
    now = time.time()
    with _SLOCK:
        items = sorted(SESSIONS.items(), key=lambda kv: kv[1].get("touched", 0))
    doomed = {}
    for sid, s in items:
        active = s.get("state") in ("queued", "running")
        if active and now - s.get("polled", now) > ABANDON_AFTER_SEC:
            doomed[sid] = "abandoned by client"
        elif not active and now - s.get("touched", now) > SESSION_TTL_SEC:
            doomed[sid] = "idle ttl"
    overflow = len(items) - len(doomed) - MAX_SESSIONS
    for sid, _ in items:
        if overflow <= 0:
            break
        if sid not in doomed:
            doomed[sid] = "session cap"
            overflow -= 1
    for sid, why in doomed.items():
        _discard_session(sid, why)

    with _SLOCK:
        uploads = list(UPLOADS.items())
    for uid, u in uploads:
        with _SLOCK:
            busy = _IN_USE.get(u["path"])
        if not busy and now - u.get("touched", now) > UPLOAD_TTL_SEC:
            _delete_upload(uid)
    _enforce_cache_cap()


def _janitor() -> None:
    while True:
        time.sleep(JANITOR_INTERVAL_SEC)
        try:
            _reap()
        except Exception:
            log.exception("janitor pass failed")


_janitor_started = False


def _ensure_janitor() -> None:
    # Started on the first request (not at import) so no thread exists before a fork.
    global _janitor_started
    if not _janitor_started:
        with _SLOCK:
            if not _janitor_started:
                threading.Thread(target=_janitor, name="janitor", daemon=True).start()
                _janitor_started = True


# --------------------------------------------------------------------------- #
# track collection -> lazy loaders + skipped report
# --------------------------------------------------------------------------- #
def _usable(path: Path, source: str, skipped: list) -> bool:
    try:
        dur = sf.info(str(path)).duration
    except Exception:
        skipped.append({"source": source, "reason": "could not decode the audio"})
        return False
    if dur < MIN_TRACK_SEC:
        skipped.append({"source": source,
                        "reason": f"too short ({dur:.0f}s, need {MIN_TRACK_SEC:.0f}s)"})
        return False
    return True


def _collect_tracks(spec: dict, s: dict, job) -> tuple:
    """Return ([(source, path)], skipped). Downloads happen here (disk-bound);
    decoding into RAM happens lazily inside build_continuous_set."""
    tracks, skipped = [], []
    kind = spec["kind"]
    if kind == "sample":
        s["status"] = "loading sample tracks"
        for f in _SAMPLE_SOURCES:
            p = ROOT / f
            if p.exists():
                tracks.append((Path(f).stem, p))
            else:
                skipped.append({"source": Path(f).stem, "reason": "sample missing"})
    elif kind == "uploads":
        s["status"] = "loading uploads"
        for uid in spec["uploads"]:
            with _SLOCK:
                u = UPLOADS.get(uid)
                if u:
                    u["touched"] = time.time()
            if not u or not Path(u["path"]).exists():
                skipped.append({"source": u["name"] if u else "upload",
                                "reason": "upload expired, please re-upload"})
                continue
            if _usable(Path(u["path"]), u["name"], skipped):
                tracks.append((u["name"], Path(u["path"])))
    else:
        s["status"] = "resolving playlist"
        vids = _resolve_playlist(spec["url"])
        for i, vid in enumerate(vids):
            job.check()
            s["status"] = f"downloading {i + 1}/{len(vids)}"
            src = f"https://youtu.be/{vid}"
            try:
                if not _enforce_cache_cap(MAX_DOWNLOAD_MB * MB):
                    raise TrackError("server storage is full")
                p = _download(vid)
            except TrackError as e:
                skipped.append({"source": src, "reason": str(e)})
                continue
            except Exception:
                log.exception("download crashed vid=%s", vid)
                skipped.append({"source": src, "reason": "download failed"})
                continue
            if _usable(p, src, skipped):
                tracks.append((src, p))
                _acquire([p])          # protect from eviction while later ones download
                s.setdefault("_held", []).append(p)
    return tracks, skipped


# --------------------------------------------------------------------------- #
# progressive producer (runs inside a JobQueue slot)
# --------------------------------------------------------------------------- #
def _produce(job, sid: str, spec: dict) -> None:
    token = _rid.set(sid)
    s = SESSIONS.get(sid)
    if s is None:
        _rid.reset(token)
        return
    t_start = time.time()
    s.update(state="running", status="starting", started=t_start)
    log.info("render start kind=%s stems=%s blend_bars=%s quick=%s queue_wait_sec=%.1f",
             spec["kind"], spec["use_stems"], spec["blend_bars"], spec.get("quick"),
             t_start - s["created"])
    CH = int(CHUNK_SEC * SR)
    buf = {"pcm": np.zeros((0, 2), dtype=np.float32), "first": None}
    held: list = []

    def _flush(final=False):
        while len(buf["pcm"]) >= CH or (final and len(buf["pcm"]) > 0):
            take = buf["pcm"][:CH]
            buf["pcm"] = buf["pcm"][CH:]
            idx = len(s["chunks"])
            p = CHUNK_DIR / f"{sid}_{idx:05d}.wav"
            # float subtype preserves >1.0 sums; 0.9 uniform safety gain (consistent
            # across chunks so no level steps).
            sf.write(p, np.clip(take * 0.9, -1.0, 1.0), SR, subtype="FLOAT")
            s["chunks"].append(str(p))
            s["touched"] = time.time()
            if buf["first"] is None:
                buf["first"] = time.time() - t_start
                log.info("first chunk after %.1fs", buf["first"])

    def _on_part(part, marker=None):
        job.check()
        buf["pcm"] = np.vstack([buf["pcm"], _as_stereo(np.asarray(part)).astype(np.float32)])
        if marker:
            s["markers"].append(marker)
        _flush(False)

    def _progress(i, n):
        job.check()
        s["status"] = f"mixing {i}/{n}"
        s["touched"] = time.time()

    def _loader(path: Path):
        def load():
            job.check()
            return _load_wav(str(path))
        return load

    try:
        tracks, skipped = _collect_tracks(spec, s, job)
        held = s.pop("_held", [])
        s["skipped"] = skipped
        if skipped:
            log.info("skipped %d track(s): %s", len(skipped),
                     "; ".join(f"{k['source']}: {k['reason']}" for k in skipped))
        if not tracks:
            raise UserError("no usable tracks — every track was skipped" if skipped
                            else "no tracks to mix")
        job.check()
        paths = [p for _, p in tracks]
        _acquire(paths)
        held += paths
        s["status"] = "mixing"
        names = [src for src, _ in tracks]

        def _on_skip(i, reason):
            s["skipped"] = list(s.get("skipped") or []) + [{"source": str(names[i]), "reason": reason}]

        _, markers = build_continuous_set(
            [_loader(p) for p in paths], sr=SR, blend_bars=spec["blend_bars"],
            use_stems=spec["use_stems"], on_part=_on_part, progress=_progress,
            play_bars=(4, 8) if spec.get("quick") else None,
            checkpoint=job.check, on_skip=_on_skip, logger=log)
        job.check()
        _flush(final=True)
        s["markers"] = markers     # authoritative full list
        dur = len(s["chunks"]) * CHUNK_SEC
        wall = time.time() - t_start
        s.update(state="ready", status="ready", n_tracks=len(tracks),
                 chunk_count=len(s["chunks"]), duration=dur, touched=time.time())
        log.info("render done tracks=%d skipped=%d audio_sec=%.1f wall_sec=%.1f "
                 "first_chunk_sec=%.1f realtime_x=%.2f", len(tracks), len(skipped), dur, wall,
                 buf["first"] or wall, dur / max(wall, 1e-6))
    except Cancelled:
        log.info("render cancelled after %.1fs", time.time() - t_start)
        s.update(state="cancelled", status="cancelled", touched=time.time())
        _delete_chunk_files(sid)
        s["chunks"] = []
    except UserError as e:
        log.warning("render error: %s (%.1fs)", e, time.time() - t_start)
        s.update(state="error", status="error", error=str(e), touched=time.time())
    except Exception:
        log.exception("render failed after %.1fs", time.time() - t_start)
        s.update(state="error", status="error", touched=time.time(),
                 error=f"the mixer failed on this set (ref {sid[:8]}) — try other tracks")
    finally:
        _release(held + s.pop("_held", []))
        if sid not in SESSIONS:              # discarded while we were still writing
            _delete_chunk_files(sid)
        _rid.reset(token)


# --------------------------------------------------------------------------- #
# request plumbing: request ids, security headers, JSON errors
# --------------------------------------------------------------------------- #
@app.before_request
def _before():
    _ensure_janitor()
    sid = (request.view_args or {}).get("sid")
    g.rid = sid if isinstance(sid, str) and _SID_RE.fullmatch(sid) else secrets.token_hex(4)
    _rid.set(g.rid)
    g.csp_nonce = secrets.token_urlsafe(16)


@app.after_request
def _security_headers(resp):
    nonce = getattr(g, "csp_nonce", "")
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
        "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        f"frame-ancestors {FRAME_ANCESTORS}")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if FRAME_ANCESTORS == "'none'":
        resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Request-ID"] = getattr(g, "rid", "-")
    if request.path.startswith("/api/") and resp.mimetype == "application/json":
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _err(msg: str, code: int, **extra):
    body = {"error": msg}
    body.update(extra)
    return jsonify(body), code


@app.errorhandler(HTTPException)
def _http_error(e):
    if e.code == 413:
        return _err(f"upload too large — max {MAX_UPLOAD_MB} MB per file, "
                    f"{MAX_PLAYLIST_TRACKS} files per request", 413)
    if e.code == 429:
        return _err(f"too many requests — limit is {e.description}", 429)
    if request.path.startswith("/api/") or request.path in ("/healthz", "/readyz"):
        return _err(e.name.lower(), e.code)
    return e


@app.errorhandler(Exception)
def _unhandled(e):
    log.exception("unhandled error on %s %s", request.method, request.path)
    return _err(f"internal error (ref {getattr(g, 'rid', '-')})", 500)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("club_player.html", stems_available=stems_available(),
                           allow_youtube=ALLOW_YOUTUBE, allow_uploads=ALLOW_UPLOADS,
                           max_upload_mb=MAX_UPLOAD_MB, max_tracks=MAX_PLAYLIST_TRACKS,
                           csp_nonce=g.csp_nonce)


@app.route("/healthz")
def healthz():
    """Liveness: the process is up and serving."""
    return jsonify({"ok": True})


def _cache_writable() -> bool:
    try:
        p = CACHE_DIR / f".rw-{secrets.token_hex(4)}"
        p.write_bytes(b"ok")
        p.unlink()
        return True
    except OSError:
        return False


@app.route("/readyz")
def readyz():
    """Readiness: tools present, cache writable, render queue has room."""
    bins = _binaries()
    q = JOBS.stats()
    checks = {"ffmpeg": bins["ffmpeg"], "ffprobe": bins["ffprobe"],
              "cache_writable": _cache_writable(), "queue_ok": not JOBS.saturated()}
    ok = all(checks.values())
    body = {"ready": ok, "checks": checks, "queue": q,
            "youtube": (bins.get("yt-dlp", False) if ALLOW_YOUTUBE else None),
            "sessions": len(SESSIONS)}
    return jsonify(body), (200 if ok else 503)


_START_KEYS = {"url", "files", "uploads", "blend_bars", "use_stems", "quick"}


def _parse_start(data) -> dict:
    """Strict validation of the /api/set/start body. Raises UserError (-> 400)."""
    if not isinstance(data, dict):
        raise UserError("body must be a JSON object")
    unknown = sorted(set(data) - _START_KEYS)
    if unknown:
        raise UserError(f"unknown field(s): {', '.join(unknown[:5])}; "
                        f"allowed: {', '.join(sorted(_START_KEYS))}")
    url, files, uploads = data.get("url"), data.get("files"), data.get("uploads")
    given = [k for k, v in (("url", url), ("files", files), ("uploads", uploads)) if v]
    if len(given) != 1:
        raise UserError("provide exactly one source: url (YouTube playlist/video), "
                        "uploads (ids from /api/upload) or files: [\"__SAMPLE__\"]")

    bb = data.get("blend_bars", 16)
    if isinstance(bb, bool) or not isinstance(bb, int) or not 8 <= bb <= 32:
        raise UserError("blend_bars must be an integer between 8 and 32")
    us = data.get("use_stems", None)
    if us is not None and not isinstance(us, bool):
        raise UserError("use_stems must be true or false")
    use_stems = (stems_available() if us is None else us) and stems_available()
    quick = data.get("quick", False)
    if not isinstance(quick, bool):
        raise UserError("quick must be true or false")
    spec = {"blend_bars": bb, "use_stems": bool(use_stems), "quick": quick}

    if url:
        if not isinstance(url, str) or len(url) > 2048:
            raise UserError("url must be a string (max 2048 chars)")
        if not ALLOW_YOUTUBE:
            raise PermissionError("YouTube ingestion is disabled on this server — upload files instead")
        url = url.strip()
        if not _is_youtube_url(url):
            raise UserError("only youtube.com / youtu.be URLs are accepted")
        spec.update(kind="youtube", url=url)
    elif files:
        if files != ["__SAMPLE__"]:
            raise UserError("arbitrary file paths are not accepted; use files: [\"__SAMPLE__\"] "
                            "or upload your tracks")
        spec.update(kind="sample")
    else:
        if not ALLOW_UPLOADS:
            raise PermissionError("uploads are disabled on this server")
        if not isinstance(uploads, list) or not all(isinstance(u, str) for u in uploads):
            raise UserError("uploads must be a list of upload id strings")
        if len(uploads) > MAX_PLAYLIST_TRACKS:
            raise UserError(f"at most {MAX_PLAYLIST_TRACKS} tracks per set")
        bad = [u for u in uploads if not _UPLOAD_ID_RE.fullmatch(u)]
        if bad:
            raise UserError("malformed upload id")
        with _SLOCK:
            missing = [u for u in uploads if u not in UPLOADS]
        if missing:
            raise UserError(f"{len(missing)} upload(s) expired or unknown — upload them again")
        spec.update(kind="uploads", uploads=list(uploads))
    return spec


@app.route("/api/set/start", methods=["POST"])
@_rate_limit
def start():
    _reap()
    if not request.is_json:
        return _err("send a JSON body with Content-Type: application/json", 400)
    data = request.get_json(silent=True)
    if data is None:
        return _err("request body is not valid JSON", 400)
    try:
        spec = _parse_start(data)
    except PermissionError as e:
        return _err(str(e), 403)
    except UserError as e:
        return _err(str(e), 400)

    sid = secrets.token_hex(16)
    g.rid = sid
    _rid.set(sid)
    now = time.time()
    with _SLOCK:
        SESSIONS[sid] = {"state": "queued", "status": "queued", "chunks": [], "markers": [],
                         "skipped": [], "chunk_sec": CHUNK_SEC, "created": now,
                         "touched": now, "polled": now, "kind": spec["kind"]}
    try:
        JOBS.submit(sid, lambda job: _produce(job, sid, spec))
    except QueueFull:
        with _SLOCK:
            SESSIONS.pop(sid, None)
        q = JOBS.stats()
        log.warning("queue full running=%d queued=%d", q["running"], q["queued"])
        resp, code = _err(f"the mixer is busy ({q['running']} rendering, {q['queued']} waiting) "
                          "— please try again in a minute", 503, retry_after=30)
        resp.headers["Retry-After"] = "30"
        return resp, code
    pos = JOBS.position(sid)
    log.info("set queued kind=%s position=%s", spec["kind"], pos)
    return jsonify({"session_id": sid, "queue_position": pos or 0,
                    "use_stems": spec["use_stems"]}), 202


def _get_session(sid: str):
    if not _SID_RE.fullmatch(sid):
        return None
    return SESSIONS.get(sid)


@app.route("/api/set/<sid>", methods=["GET"])
def status(sid):
    s = _get_session(sid)
    if not s:
        return _err("set not found (it may have expired)", 404)
    s["touched"] = s["polled"] = time.time()
    pos = JOBS.position(sid) if s.get("state") == "queued" else None
    return jsonify({
        "state": s.get("state"),
        "status": s.get("status"),
        "error": s.get("error"),
        "queue_position": pos,
        "skipped": s.get("skipped") or [],
        "chunk_sec": s.get("chunk_sec", CHUNK_SEC),
        "chunks_ready": len(s.get("chunks", [])),      # grows during render
        "chunk_count": s.get("chunk_count"),           # set only when finished
        "duration": s.get("duration"),
        "n_tracks": s.get("n_tracks"),
        "markers": s.get("markers") or None,           # live, grows during render
    })


@app.route("/api/set/<sid>", methods=["DELETE"])
@app.route("/api/set/<sid>/cancel", methods=["POST"])     # navigator.sendBeacon form
def cancel(sid):
    if not _SID_RE.fullmatch(sid) or not _discard_session(sid, "cancelled by client"):
        return _err("set not found", 404)
    return jsonify({"cancelled": True})


@app.route("/api/set/<sid>/chunk/<int:n>")
def chunk(sid, n):
    s = _get_session(sid)
    if not s:
        return _err("set not found (it may have expired)", 404)
    s["touched"] = s["polled"] = time.time()
    chunks = s.get("chunks", [])
    if n >= len(chunks):
        # not produced yet (client should retry) or past the end
        code = 416 if s.get("state") in ("ready", "error", "cancelled") else 404
        return ("", code)
    try:
        return send_file(chunks[n], mimetype="audio/wav", conditional=True, max_age=3600)
    except FileNotFoundError:
        return _err("chunk no longer available", 410)


@app.route("/api/upload", methods=["POST"])
@_upload_limit
def upload():
    """Multipart upload of one or more audio files (field name `file`). Each file is
    content-sniffed with ffprobe, decoded to WAV at the server rate (capped at
    PER_TRACK_CAP_SEC) and registered under a random id for /api/set/start."""
    if not ALLOW_UPLOADS:
        return _err("uploads are disabled on this server", 403)
    files = [f for f in request.files.getlist("file") if f and f.filename is not None]
    if not files:
        return _err("send audio as multipart/form-data in the field 'file'", 400)
    if len(files) > MAX_PLAYLIST_TRACKS:
        return _err(f"at most {MAX_PLAYLIST_TRACKS} files per request", 400)
    _reap()
    if not _enforce_cache_cap(request.content_length or 0):
        log.warning("upload refused: cache over %.1f GB", MAX_CACHE_GB)
        return _err("server storage is full — try again later", 507)

    accepted, rejected = [], []
    for f in files:
        name = _clean_name(f.filename)
        uid = secrets.token_hex(16)
        raw = UPLOAD_DIR / f"{uid}.upload"
        dst = UPLOAD_DIR / f"{uid}.wav"
        try:
            f.stream.seek(0, os.SEEK_END)
            size = f.stream.tell()
            f.stream.seek(0)
            if size > MAX_UPLOAD_MB * MB:
                raise TrackError(f"larger than {MAX_UPLOAD_MB} MB")
            if size == 0:
                raise TrackError("empty file")
            f.save(str(raw))
            _probe_audio(raw)
            dur = _decode_to_wav(raw, dst)
            if dur < MIN_TRACK_SEC:
                dst.unlink(missing_ok=True)
                raise TrackError(f"too short ({dur:.0f}s, need {MIN_TRACK_SEC:.0f}s)")
            now = time.time()
            with _SLOCK:
                UPLOADS[uid] = {"path": str(dst), "name": name, "duration": dur,
                                "created": now, "touched": now}
            accepted.append({"id": uid, "name": name, "duration": round(dur, 1)})
            log.info("upload ok id=%s bytes=%d duration=%.1f", uid[:8], size, dur)
        except TrackError as e:
            rejected.append({"name": name, "reason": str(e)})
            log.info("upload rejected: %s", e)
        except Exception:
            log.exception("upload failed")
            rejected.append({"name": name, "reason": "could not process this file"})
        finally:
            raw.unlink(missing_ok=True)
    if not accepted:
        return _err(rejected[0]["reason"] if len(rejected) == 1 else "no usable audio files",
                    415 if all(r["reason"] == "not a recognised audio file" for r in rejected)
                    else 400, uploads=[], rejected=rejected)
    return jsonify({"uploads": accepted, "rejected": rejected})


@app.route("/api/upload/<uid>", methods=["DELETE"])
def delete_upload(uid):
    if not _UPLOAD_ID_RE.fullmatch(uid) or uid not in UPLOADS:
        return _err("upload not found", 404)
    with _SLOCK:
        busy = _IN_USE.get(UPLOADS[uid]["path"])
    if not busy:
        _delete_upload(uid)
    return jsonify({"deleted": True})


# --------------------------------------------------------------------------- #
# startup + shutdown
# --------------------------------------------------------------------------- #
def _startup() -> None:
    # Session/upload registries are in memory, so any chunk or upload on disk at
    # boot belongs to a previous process: delete it. Partial downloads too.
    removed = 0
    for pattern_dir, pattern in ((CHUNK_DIR, "*"), (UPLOAD_DIR, "*"),
                                 (DL_DIR, "*.src.*"), (DL_DIR, "*.part.wav")):
        for p in pattern_dir.glob(pattern):
            try:
                if p.is_file():
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    if removed:
        log.info("startup: removed %d stale cache files", removed)
    bins = _binaries()
    for name, present in bins.items():
        if not present:
            log.error("startup: '%s' not found on PATH — %s", name,
                      "YouTube sets will fail (set ALLOW_YOUTUBE=false to hide it)"
                      if name == "yt-dlp" else "uploads and downloads cannot be decoded")
    log.info("startup: renders=%d queue=%d uploads=%s youtube=%s stems=%s cache=%s cap=%.1fGB",
             MAX_CONCURRENT_RENDERS, MAX_QUEUE_LENGTH, ALLOW_UPLOADS, ALLOW_YOUTUBE,
             stems_available(), CACHE_DIR, MAX_CACHE_GB)
    try:
        ensure_demo_samples(DEMO_DIR, sr=SR)   # royalty-free demo loops (idempotent)
    except Exception:
        log.exception("startup: could not generate demo samples")
    _enforce_cache_cap()


def shutdown() -> None:
    """Cancel queued/running renders and delete their chunks (gunicorn worker_exit)."""
    JOBS.shutdown()
    with _SLOCK:
        sids = list(SESSIONS)
    for sid in sids:
        _discard_session(sid, "shutdown")


_startup()

if __name__ == "__main__":
    log.info("Club AutoMixer on http://%s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
