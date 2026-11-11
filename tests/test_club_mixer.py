"""Behavioural tests for the club mixer — the things a listener would notice.

Run with:  python -m pytest tests/ -q      (or:  python tests/test_club_mixer.py)

All audio is synthesized (src/demo_samples.py), so the true tempo, beat positions
and song structure are known exactly and the mixer can be held to them.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import club_mixer as cm                      # noqa: E402
from src import dj_techniques as dj                   # noqa: E402
from src import technique_bridge as bridge            # noqa: E402
from src.demo_samples import _synth_track             # noqa: E402

SR = 44100
_CACHE = {}


def track(bpm, root):
    if (bpm, root) not in _CACHE:
        _CACHE[(bpm, root)] = _synth_track(bpm, root, SR)
    return _CACHE[(bpm, root)]


def pad_track(seconds=60):
    """Beatless ambient pad — nothing to beat-match."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
            + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def kick_lag_ms(a, b, maxlag=160):
    """Offset (ms) between the kick patterns of two stereo segments."""
    sos = sig.butter(4, 150 / (SR / 2), btype="low", output="sos")
    ea = cm._amp_onset_env(sig.sosfilt(sos, a.mean(axis=1)).astype(np.float32), 32)
    eb = cm._amp_onset_env(sig.sosfilt(sos, b.mean(axis=1)).astype(np.float32), 32)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    best, bl = -np.inf, 0
    for lag in range(-maxlag, maxlag + 1):
        v = float(np.dot(ea[lag:], eb[:len(eb) - lag])) if lag >= 0 else float(np.dot(ea[:lag], eb[-lag:]))
        if v > best:
            best, bl = v, lag
    return bl * 32 / SR * 1000


def capture_overlaps():
    """Spy on the engine to get the (A, B) deck pairs it mixes, once both have been
    warped onto the shared beat grid."""
    got, orig = [], cm.hole_db

    def spy(out, a, b, sr):
        got.append((a.copy(), b.copy()))
        return orig(out, a, b, sr)
    cm.hole_db = spy
    return got, lambda: setattr(cm, "hole_db", orig)


def fake_stems(seg):
    """Four 'stems' that sum to the segment (band splits) — exercises the stem
    techniques without needing demucs."""
    low, mid, high = dj.band_split(seg, SR)
    return {"drums": (low * 0.5 + high).astype(np.float32), "bass": (low * 0.5).astype(np.float32),
            "other": (mid * 0.5).astype(np.float32), "vocals": (mid * 0.5).astype(np.float32)}


# --------------------------------------------------------------------------- #
def test_beat_grid_is_accurate():
    for bpm, root in ((122.0, 45), (128.0, 42), (98.0, 43)):
        g = cm.build_phrase_grid(track(bpm, root), SR)
        assert abs(g.tempo - bpm) < 0.05, (bpm, g.tempo)
        period = 60.0 / bpm
        beats = g.bt[(g.bk >= 0) & (g.bt < g.duration - 1)]
        err = (beats / period - np.round(beats / period)) * period
        # a few ms of constant analysis latency is fine (identical for every track,
        # so it cancels when two tracks are aligned) — jitter and drift are not
        assert np.max(np.abs(err)) < 0.008, f"{bpm}: beats up to {np.max(np.abs(err))*1000:.1f} ms off the kick"
        assert np.std(err) < 0.0015, f"{bpm}: beat jitter {np.std(err)*1000:.2f} ms"
        # downbeats on bar lines, sections on the 8-bar phrase grid
        assert np.max(np.abs((g.downbeats / (4 * period)) - np.round(g.downbeats / (4 * period)))) < 0.02
        assert {8, 16, 32, 40, 56} <= {i for i, _, _ in g.boundaries}
        assert g.beat_conf > cm.BEAT_CONF_MIN


def test_beatless_audio_is_not_trusted():
    assert cm.build_phrase_grid(pad_track(), SR).beat_conf < cm.BEAT_CONF_MIN


def test_key_detection_and_compatibility():
    a = cm.build_phrase_grid(track(122.0, 45), SR)     # A minor
    b = cm.build_phrase_grid(track(124.0, 40), SR)     # E minor  (one step away)
    c = cm.build_phrase_grid(track(128.0, 42), SR)     # F# minor (clashes with E minor)
    assert a.camelot[:-1] == "8" and b.camelot[:-1] == "9" and c.camelot[:-1] == "11"
    assert cm.keys_compatible(a, b) and not cm.keys_compatible(b, c)


def test_wsola_hands_back_to_original_audio_seamlessly():
    x = cm._as_stereo(track(124.0, 40))
    H, J = 1024, 400
    pos = 5 * SR + np.arange(J) * H * 0.96             # play 4% slow
    out, nxt = cm._wsola(x, pos)
    assert len(out) == J * H
    assert abs(nxt - (pos[-1] + H)) < 1200              # ended where the map said (±~25 ms)
    joined = np.vstack([out, x[nxt:nxt + 4096]])
    seam = np.abs(np.diff(joined[:, 0]))[J * H - 3:J * H + 3].max()
    assert seam <= np.abs(np.diff(x[:, 0])).max()       # no click at the handoff
    assert np.allclose(out[J * H - 8:], x[nxt - 8:nxt], atol=2e-3)


def test_move_family_follows_what_a_dj_would_do():
    def plan(a, b, **kw):
        return cm.plan_transition(cm.build_phrase_grid(a, SR), cm.build_phrase_grid(b, SR), **kw)
    blendable = set(bridge.EXECUTOR_FX) | set(dj.EQ_TECHNIQUES)
    p = plan(track(122.0, 45), track(124.0, 40))
    assert p.family == "beatmatch" and p.transition_type in blendable and p.key_compatible
    p = plan(track(124.0, 40), track(128.0, 42))                    # key clash: nothing melodic overlaps
    assert p.family == "beatmatch" and p.transition_type in {"swap", "drop_on_the_one", "quick_cut",
                                                             "back_and_forth", "loop_transition",
                                                             "drum_roll", "chop", "double_drop", "drop_swap"}
    p = plan(track(122.0, 45), track(98.0, 43))                     # 24% apart: an effect cut, not a blend
    assert p.family == "cut" and p.transition_type in set(bridge.ORIGINAL_CUTS) | set(dj.CUT_TECHNIQUES)
    assert plan(track(122.0, 45), pad_track()).family == "fade"     # no beat at all
    # a tempo gap inside reach is beat-matched (the tempo glides), not cut
    assert plan(track(122.0, 45), track(134.0, 40)).family == "beatmatch"
    # never the same move twice in a row
    first = plan(track(122.0, 45), track(124.0, 40)).transition_type
    assert plan(track(122.0, 45), track(124.0, 40), recent_types=[first]).transition_type != first


def test_blend_is_phrase_aligned():
    a, b = track(122.0, 45), track(124.0, 40)
    p = cm.plan_transition(cm.build_phrase_grid(a, SR), cm.build_phrase_grid(b, SR), blend_bars=16)
    bar_a, bar_b = 4 * 60 / 122.0, 4 * 60 / 124.0
    assert (p.pre_bars, p.post_bars) == (8, 8)
    assert abs(p.a_switch_sec - 56 * bar_a) < 0.02      # A leaves where its outro starts
    assert abs(p.b_switch_sec - 8 * bar_b) < 0.02       # B's intro ends exactly on the switch
    assert abs(p.b_in_sec) < 0.02


def test_kicks_stay_locked_through_the_blend():
    got, restore = capture_overlaps()
    try:
        cm.build_continuous_set([track(122.0, 45), track(124.0, 40), track(128.0, 42)], sr=SR,
                                use_stems=False)
    finally:
        restore()
    assert len(got) >= 2
    for a, b in got:
        win = 4 * SR
        lags = [kick_lag_ms(a[i:i + win], b[i:i + win]) for i in range(0, len(a) - win + 1, win)]
        assert max(abs(v) for v in lags) < 12.0, f"kicks drift {lags} ms"


def test_every_original_technique_runs_and_joins_cleanly():
    """The project's original library, run through the bridge on two aligned decks:
    each technique must render, stay finite, begin as pure A and end as pure B."""
    got, restore = capture_overlaps()
    try:
        cm.build_continuous_set([track(122.0, 45), track(124.0, 40)], sr=SR, use_stems=False)
    finally:
        restore()
    a, b = got[0]
    sa, sb = fake_stems(a), fake_stems(b)
    h = int(0.02 * SR)

    def rel(x, y):
        return float(np.sqrt(np.mean((x - y) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    for name in bridge.ORIGINAL:
        if name == "progressive_morph":
            continue                                   # slow spectral morph; covered by the stems run
        out = bridge.run_original(name, a, b, SR, 122.0, 124.0, stems_a=sa, stems_b=sb,
                                  beat=int(60 / 122.0 * SR))
        assert out.shape == a.shape and np.isfinite(out).all(), name
        assert rel(out[:h], a[:h]) < 0.08, f"{name} does not start as A"
        assert rel(out[-h:], b[-h:]) < 0.08, f"{name} does not end as B"
        assert dj.hole_db(out, a, b, SR) > -20.0, f"{name} leaves dead air"


def test_original_technique_repairs():
    from src.technique_executor import TechniqueExecutor
    ex = TechniqueExecutor(sr=SR)
    n = 8 * SR
    a = np.full((n, 2), 0.5, dtype=np.float32)
    b = np.full((n, 2), -0.25, dtype=np.float32)
    for name in ("quick_cut", "drop_on_the_one", "back_and_forth"):   # B used to never arrive
        out = ex.execute(name, a.copy(), b.copy(), {"tempo": 120.0})
        assert np.allclose(out[:SR], 0.5, atol=1e-3) and np.allclose(out[-SR:], -0.25, atol=1e-3), name
    out = ex.execute("energy_build", a.copy(), b.copy(), {"filter_sweep": False, "eq_boost": False})
    assert abs(out[-1, 0] + 0.25) < 1e-3               # B used to fade OUT (ended at 0)
    # curve smoothing: the fast version matches scipy
    from scipy.ndimage import gaussian_filter1d as ref
    from src.curve_smooth import gaussian_filter1d as fast
    c = np.clip(np.linspace(-1, 2, 300000), 0, 1)
    assert np.max(np.abs(fast(c, 2000) - ref(c, 2000))) < 2e-3


def test_set_is_contiguous_level_and_returns_to_native_tempo():
    tracks = [track(122.0, 45), 0.4 * track(124.0, 40), track(128.0, 42)]   # middle one mastered quiet
    audio, markers = cm.build_continuous_set(tracks, sr=SR, use_stems=False)
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 0.971
    # markers tile the timeline exactly
    assert markers[0]["start_sec"] == 0.0
    for m0, m1 in zip(markers, markers[1:]):
        assert abs(m0["end_sec"] - m1["start_sec"]) < 1e-6
    assert abs(markers[-1]["end_sec"] - len(audio) / SR) < 1e-3
    # nearly all of every song is played (no mixing out half-way through)
    assert len(audio) / SR > 0.8 * sum(len(t) for t in tracks) / SR

    def rms_db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    bodies = [m for m in markers if m["type"] == "track"]
    levels = [rms_db(audio[int(m["start_sec"] * SR):int(m["end_sec"] * SR)]) for m in bodies]
    assert max(levels) - min(levels) < 1.5, f"track levels differ: {levels}"
    for m in (m for m in markers if m["type"] == "transition"):
        e = int(m["end_sec"] * SR)
        assert abs(rms_db(audio[e:e + SR]) - rms_db(audio[e - SR:e])) < 2.0   # no jump when A is gone
    # each track is back at its own tempo by the end of its body
    for m, bpm in zip(bodies, (122.0, 124.0, 128.0)):
        e = int(m["end_sec"] * SR)
        g = cm._analyze(audio[e - 30 * SR:e].mean(axis=1), SR, 8)
        assert abs(g.tempo - bpm) < 0.3, (bpm, g.tempo)


def test_streaming_emits_the_same_timeline():
    tracks = [track(122.0, 45), track(98.0, 43)]
    full, markers = cm.build_continuous_set(tracks, sr=SR, use_stems=False)
    parts, live = [], []
    _, markers2 = cm.build_continuous_set(tracks, sr=SR, use_stems=False, on_part=lambda p, m=None: (parts.append(p), m and live.append(m)))
    stream = np.vstack(parts)
    assert len(stream) == len(full) and np.allclose(stream, full, atol=1e-6)
    assert [m["type"] for m in live] == [m["type"] for m in markers] == [m["type"] for m in markers2]


def test_two_track_api_and_tiny_inputs():
    out, info = cm.render_club_mix(track(122.0, 45), track(124.0, 40), sr=SR, use_stems=False)
    assert info["family"] == "beatmatch" and out.ndim == 2 and len(out) > 30 * SR
    # degenerate inputs must not crash
    short = track(122.0, 45)[:5 * SR]
    audio, markers = cm.build_continuous_set([short, short, np.zeros(3 * SR, dtype=np.float32)],
                                             sr=SR, use_stems=False)
    assert np.isfinite(audio).all() and len(markers) == 5
    assert cm.build_continuous_set([], sr=SR)[1] == []


def test_engine_hooks_cancel_and_skip():
    class Stop(BaseException):
        pass
    calls = {"n": 0}

    def checkpoint():
        calls["n"] += 1
        if calls["n"] > 3:
            raise Stop()
    try:
        cm.build_continuous_set([track(122.0, 45), track(124.0, 40)], sr=SR, use_stems=False,
                                checkpoint=checkpoint)
        assert False, "checkpoint did not cancel the render"
    except Stop:
        pass
    # a broken track is skipped and reported; the set carries on with the next one
    skipped = []

    def broken():
        raise IOError("disk on fire")
    audio, markers = cm.build_continuous_set([track(122.0, 45), broken, track(124.0, 40)], sr=SR,
                                             use_stems=False, on_skip=lambda i, r: skipped.append((i, r)))
    assert [i for i, _ in skipped] == [1] and "disk on fire" in skipped[0][1]
    assert [m["index"] for m in markers if m["type"] == "track"] == [0, 2]
    t = [m for m in markers if m["type"] == "transition"][0]
    assert (t["from"], t["to"]) == (0, 2) and np.isfinite(audio).all()


def test_quick_mix_mode_mixes_out_early_without_cutting():
    tracks = [track(122.0, 45), track(124.0, 40), track(128.0, 42)]
    full, mf = cm.build_continuous_set(tracks, sr=SR, use_stems=False)
    quick, mq = cm.build_continuous_set(tracks, sr=SR, use_stems=False, play_bars=(4, 8))
    assert len(quick) < 0.8 * len(full)
    for m0, m1 in zip(mq, mq[1:]):                      # still one continuous timeline
        assert abs(m0["end_sec"] - m1["start_sec"]) < 1e-6
    # the middle track plays on its own for 4-8 bars between its two transitions
    mid = [m for m in mq if m["type"] == "track" and m["index"] == 1][0]
    bar = 4 * 60 / 124.0
    solo_bars = (mid["end_sec"] - mid["start_sec"]) / bar
    assert 3.5 <= solo_bars <= 8.5, solo_bars
    assert all(m["family"] == "beatmatch" for m in mq if m["type"] == "transition")


def test_stems_path_end_to_end():
    """With demucs installed the engine separates each overlap and picks a stem move."""
    if not cm.stems_available():
        print("     (skipped: demucs not installed)")
        return
    audio, markers = cm.build_continuous_set([track(122.0, 45), track(124.0, 40)], sr=SR, use_stems=True)
    t = [m for m in markers if m["type"] == "transition"][0]
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 0.971
    assert t["stems"] and t["family"] == "beatmatch"
    assert t["transition_type"] in set(bridge.ORIGINAL) | set(dj.STEM_TECHNIQUES) | {"chop"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
