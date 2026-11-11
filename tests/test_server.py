"""
Server tests (Flask test client, no network).

    ./.venv/bin/python -m pytest tests/test_server.py -q

Env is set BEFORE importing club_server because src.settings reads it at import.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_TMP = Path(tempfile.mkdtemp(prefix="automix-test-"))
os.environ.update({
    "CACHE_DIR": str(_TMP / "cache"),
    "RATE_LIMIT": "10000 per hour",
    "UPLOAD_RATE_LIMIT": "10000 per hour",
    "MIN_TRACK_SEC": "1",
    "MAX_UPLOAD_MB": "2",
    "MAX_PLAYLIST_TRACKS": "4",
    "ALLOW_YOUTUBE": "true",
    "ALLOW_UPLOADS": "true",
})
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import club_server as cs  # noqa: E402
from src.jobs import JobQueue  # noqa: E402


# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    return cs.app.test_client()


@pytest.fixture
def small_queue(monkeypatch):
    """1 render slot + 1 waiting place, with a fake engine that streams forever
    (until cancelled) so the slot stays busy deterministically."""
    q = JobQueue(workers=1, max_queue=1, name="test")
    monkeypatch.setattr(cs, "JOBS", q)
    entered = threading.Event()

    def fake_build(loaders, sr, blend_bars, use_stems, on_part, progress, **_hooks):
        entered.set()
        part = np.zeros((int(0.5 * sr), 2), dtype=np.float32)
        for _ in range(2000):              # ~ forever; on_part raises once cancelled
            on_part(part, None)
            time.sleep(0.01)
        raise AssertionError("fake render was never cancelled")

    monkeypatch.setattr(cs, "build_continuous_set", fake_build)
    yield q, entered
    q.shutdown()


def _wav_bytes(seconds=2.0, sr=22050):
    import io
    t = np.arange(int(seconds * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def _upload(client, data, name):
    return client.post("/api/upload", data={"file": (data, name)},
                       content_type="multipart/form-data")


def _wait_state(client, sid, states, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/set/{sid}").get_json()
        if j.get("state") in states:
            return j
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {states}; last={j}")


# --------------------------------------------------------------------------- #
# health / headers
# --------------------------------------------------------------------------- #
def test_healthz_and_readyz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    r = client.get("/readyz")
    j = r.get_json()
    assert r.status_code == 200, j
    assert j["ready"] and j["checks"]["ffmpeg"] and j["checks"]["cache_writable"]


def test_readyz_fails_without_ffmpeg(client, monkeypatch):
    monkeypatch.setattr(cs, "_which", lambda b: None)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.get_json()["checks"]["ffmpeg"] is False


def test_security_headers(client):
    for path in ("/", "/healthz", "/api/set/" + "0" * 32):
        r = client.get(path)
        csp = r.headers.get("Content-Security-Policy", "")
        assert "frame-ancestors 'none'" in csp and "default-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["Referrer-Policy"] == "no-referrer"
        assert r.headers.get("X-Request-ID")


def test_index_uses_csp_nonce(client):
    r = client.get("/")
    html = r.get_data(as_text=True)
    nonce = r.headers["Content-Security-Policy"].split("'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">' in html and f'<style nonce="{nonce}">' in html
    assert 'style="' not in html            # inline style attributes would be CSP-blocked
    assert 'id="drop"' in html and 'id="retry"' in html


# --------------------------------------------------------------------------- #
# validation / SSRF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body,needle", [
    ({}, "exactly one source"),
    ({"url": "https://youtu.be/dQw4w9WgXcQ", "files": ["__SAMPLE__"]}, "exactly one source"),
    ({"files": ["__SAMPLE__"], "blend_bars": 64}, "blend_bars"),
    ({"files": ["__SAMPLE__"], "blend_bars": "16"}, "blend_bars"),
    ({"files": ["__SAMPLE__"], "use_stems": "yes"}, "use_stems"),
    ({"files": ["__SAMPLE__"], "evil": 1}, "unknown field"),
    ({"files": ["/etc/passwd"]}, "arbitrary file paths"),
    ({"uploads": ["../../x"]}, "malformed upload id"),
    ({"uploads": ["a" * 32]}, "expired or unknown"),
    ({"uploads": ["a" * 32] * 9}, "at most"),
    ({"url": 42}, "url must be a string"),
])
def test_start_validation(client, body, needle):
    r = client.post("/api/set/start", json=body)
    assert r.status_code == 400, r.get_json()
    assert needle in r.get_json()["error"]


def test_start_rejects_non_json(client):
    r = client.post("/api/set/start", data="url=x", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 400 and "JSON" in r.get_json()["error"]
    r = client.post("/api/set/start", data="{not json", content_type="application/json")
    assert r.status_code == 400 and "not valid JSON" in r.get_json()["error"]
    r = client.post("/api/set/start", json=["files"])
    assert r.status_code == 400 and "JSON object" in r.get_json()["error"]


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost:5005/api/set/start",
    "file:///etc/passwd",
    "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
    "https://evil.example/?u=https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "ftp://www.youtube.com/x",
])
def test_ssrf_rejected(client, url):
    r = client.post("/api/set/start", json={"url": url})
    assert r.status_code == 400
    assert "youtube" in r.get_json()["error"]


def test_unknown_session_404(client):
    assert client.get("/api/set/" + "f" * 32).status_code == 404
    assert client.get("/api/set/../../etc").status_code == 404
    assert client.delete("/api/set/" + "f" * 32).status_code == 404


# --------------------------------------------------------------------------- #
# uploads
# --------------------------------------------------------------------------- #
def test_upload_wav_accepted(client):
    r = _upload(client, _wav_bytes(), "my track.wav")
    j = r.get_json()
    assert r.status_code == 200, j
    up = j["uploads"][0]
    assert up["name"] == "my track.wav" and abs(up["duration"] - 2.0) < 0.1
    stored = Path(cs.UPLOADS[up["id"]]["path"])
    assert stored.parent == cs.UPLOAD_DIR and up["id"] in stored.name and "my track" not in stored.name
    info = sf.info(str(stored))
    assert info.samplerate == cs.SR and info.channels == 2
    assert not list(cs.UPLOAD_DIR.glob("*.upload"))          # raw upload deleted after decode


def test_upload_text_renamed_mp3_rejected(client):
    import io
    before = set(cs.UPLOAD_DIR.iterdir())
    r = _upload(client, io.BytesIO(b"this is not audio, just some text\n" * 200), "song.mp3")
    j = r.get_json()
    assert r.status_code in (400, 415), j
    assert j["rejected"][0]["reason"] == "not a recognised audio file"
    assert j["uploads"] == []
    assert set(cs.UPLOAD_DIR.iterdir()) == before               # nothing left behind


def test_upload_too_large_rejected(client):
    import io
    big = io.BytesIO(b"\0" * (3 * 1024 * 1024))              # MAX_UPLOAD_MB=2
    r = _upload(client, big, "big.wav")
    assert r.status_code == 400
    assert "larger than 2 MB" in r.get_json()["rejected"][0]["reason"]


def test_upload_requires_file_field(client):
    r = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "'file'" in r.get_json()["error"]


def test_upload_count_cap(client):
    files = [(_wav_bytes(0.5), f"t{i}.wav") for i in range(5)]   # MAX_PLAYLIST_TRACKS=4
    r = client.post("/api/upload", data={"file": files}, content_type="multipart/form-data")
    assert r.status_code == 400 and "at most 4" in r.get_json()["error"]


def test_errors_never_leak_paths(client):
    import io
    r = _upload(client, io.BytesIO(b"garbage" * 100), "x.flac")
    assert str(cs.CACHE_DIR) not in r.get_data(as_text=True)
    assert "/" not in r.get_json()["rejected"][0]["reason"]


# --------------------------------------------------------------------------- #
# queue / cancellation
# --------------------------------------------------------------------------- #
def test_queue_full_returns_503(client, small_queue):
    q, entered = small_queue
    a = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    assert a.status_code == 202
    assert entered.wait(10)
    b = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    assert b.status_code == 202 and b.get_json()["queue_position"] == 1
    sb = client.get(f"/api/set/{b.get_json()['session_id']}").get_json()
    assert sb["state"] == "queued" and sb["queue_position"] == 1
    c = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    assert c.status_code == 503
    assert "busy" in c.get_json()["error"] and c.headers.get("Retry-After")
    assert client.get("/readyz").status_code == 503
    for r in (a, b):
        client.delete(f"/api/set/{r.get_json()['session_id']}")


def test_cancel_frees_slot_and_deletes_chunks(client, small_queue):
    q, entered = small_queue
    a = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    sid = a.get_json()["session_id"]
    assert entered.wait(10)
    _wait_until(lambda: client.get(f"/api/set/{sid}").get_json()["chunks_ready"] >= 2)
    assert list(cs.CHUNK_DIR.glob(f"{sid}_*.wav"))

    # a queued job is cancelled immediately and leaves the line
    b = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    bsid = b.get_json()["session_id"]
    assert client.delete(f"/api/set/{bsid}").get_json() == {"cancelled": True}
    assert q.stats()["queued"] == 0

    # the running one unwinds from inside on_part, frees the slot, deletes its chunks
    assert client.delete(f"/api/set/{sid}").status_code == 200
    assert q.wait_idle(sid, timeout=10)
    assert q.stats() == {"workers": 1, "running": 0, "queued": 0, "max_queue": 1}
    assert not list(cs.CHUNK_DIR.glob(f"{sid}_*.wav"))
    assert client.get(f"/api/set/{sid}").status_code == 404

    # ...so a new set starts right away (sendBeacon form of cancel works too)
    entered.clear()
    c = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    assert c.status_code == 202 and c.get_json()["queue_position"] == 0
    assert entered.wait(10)
    assert client.post(f"/api/set/{c.get_json()['session_id']}/cancel").status_code == 200
    assert q.wait_idle(c.get_json()["session_id"], timeout=10)


def _wait_until(pred, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met")


# --------------------------------------------------------------------------- #
# skipped tracks / no usable tracks
# --------------------------------------------------------------------------- #
def test_uploads_set_with_missing_track_skips_it(client, monkeypatch):
    """A track that vanished is reported in `skipped`; a set with none left errors cleanly."""
    up = _upload(client, _wav_bytes(), "gone.wav").get_json()["uploads"][0]
    Path(cs.UPLOADS[up["id"]]["path"]).unlink()
    r = client.post("/api/set/start", json={"uploads": [up["id"]], "use_stems": False})
    assert r.status_code == 202
    j = _wait_state(client, r.get_json()["session_id"], ("error", "ready"))
    assert j["state"] == "error" and "no usable tracks" in j["error"]
    assert j["skipped"] == [{"source": "gone.wav", "reason": "upload expired, please re-upload"}]


def test_engine_crash_is_reported_without_internals(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(f"secret detail at {cs.CACHE_DIR}/x.wav")
    monkeypatch.setattr(cs, "build_continuous_set", boom)
    r = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False})
    j = _wait_state(client, r.get_json()["session_id"], ("error", "ready"))
    assert j["state"] == "error"
    assert "secret" not in j["error"] and str(cs.CACHE_DIR) not in j["error"]


# --------------------------------------------------------------------------- #
# full renders (real engine, no stems)
# --------------------------------------------------------------------------- #
def test_sample_render_reaches_ready(client):
    r = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "use_stems": False,
                                            "blend_bars": 16})
    assert r.status_code == 202, r.get_json()
    sid = r.get_json()["session_id"]
    j = _wait_state(client, sid, ("ready", "error"), timeout=120)
    assert j["state"] == "ready", j
    assert j["n_tracks"] == 3 and j["skipped"] == []
    assert j["chunk_count"] == j["chunks_ready"] > 0
    trans = [m for m in j["markers"] if m["type"] == "transition"]
    assert len(trans) == 2
    for m in trans:
        for k in ("label", "description", "library", "stems", "bpm_from", "bpm_to",
                  "key_from", "key_to", "reason"):
            assert k in m, k
    c0 = client.get(f"/api/set/{sid}/chunk/0")
    assert c0.status_code == 200 and c0.mimetype == "audio/wav" and len(c0.data) > 1000
    assert client.get(f"/api/set/{sid}/chunk/{j['chunk_count']}").status_code == 416
    assert client.delete(f"/api/set/{sid}").status_code == 200
    assert not list(cs.CHUNK_DIR.glob(f"{sid}_*.wav"))


def test_upload_render_reaches_ready(client):
    ids = [_upload(client, _wav_bytes(20.0), f"u{i}.wav").get_json()["uploads"][0]["id"]
           for i in range(2)]
    r = client.post("/api/set/start", json={"uploads": ids, "use_stems": False, "blend_bars": 8})
    j = _wait_state(client, r.get_json()["session_id"], ("ready", "error"), timeout=120)
    # A sine has no beat grid; the engine may or may not like it. Either way the
    # server must report a clean, path-free outcome.
    assert j["state"] in ("ready", "error"), j
    if j["state"] == "error":
        assert str(cs.CACHE_DIR) not in j["error"]
    else:
        assert j["n_tracks"] == 2


# --------------------------------------------------------------------------- #
# reaper / cache
# --------------------------------------------------------------------------- #
def test_reaper_expires_uploads_and_cache_cap_evicts(client, monkeypatch):
    up = _upload(client, _wav_bytes(), "old.wav").get_json()["uploads"][0]
    path = Path(cs.UPLOADS[up["id"]]["path"])
    cs.UPLOADS[up["id"]]["touched"] = time.time() - cs.UPLOAD_TTL_SEC - 5
    cs._reap()
    assert up["id"] not in cs.UPLOADS and not path.exists()

    base = cs._dir_size(cs.CACHE_DIR)                        # whatever other tests left
    old = cs.DL_DIR / "aaaaaaaaaaa_150.wav"
    new = cs.DL_DIR / "bbbbbbbbbbb_150.wav"
    old.write_bytes(b"\0" * 400_000)
    new.write_bytes(b"\0" * 400_000)
    os.utime(old, (time.time() - 100, time.time() - 100))
    monkeypatch.setattr(cs, "MAX_CACHE_GB", (base + 600_000) / 1024 ** 3)   # room for one
    assert cs._enforce_cache_cap() is True
    assert not old.exists() and new.exists()                 # oldest evicted first
    new.unlink()


def test_startup_cleans_stale_files():
    stale = [cs.CHUNK_DIR / "deadbeef_00000.wav", cs.UPLOAD_DIR / "x.upload",
             cs.DL_DIR / "abcdefghijk.src.webm"]
    for p in stale:
        p.write_bytes(b"x")
    keep = cs.DL_DIR / "abcdefghijk_150.wav"
    keep.write_bytes(b"x")
    cs._startup()
    assert not any(p.exists() for p in stale) and keep.exists()
    keep.unlink()


def test_quick_flag_validated(client):
    r = client.post("/api/set/start", json={"files": ["__SAMPLE__"], "quick": "yes"})
    assert r.status_code == 400 and "quick" in r.get_json()["error"]
