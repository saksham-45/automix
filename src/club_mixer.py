"""
Club mixer — the "DJ brain" behind automix.

The engine does what a working DJ does, in the order they do it:

  LISTEN   Analyse each track once: a beat grid accurate to a few ms (comb-folded
           period, kick-anchored phase), downbeats, the 8-bar phrase grid, section
           boundaries (intro end / outro start), musical key (Camelot), integrated
           loudness and how trustworthy the beat grid is.

  DECIDE   Where to leave A (a phrase line, ideally where its outro starts), where
           to cue B (so the end of its intro lands on the switch), and which FAMILY
           of move fits the pair:
             beat-matched  tempos within reach (incl. half/double time). With demucs
                           stems the technique is picked from what is actually
                           playing — who is singing, who has drums — out of
                           stem_blend / layered_reveal / beat_swap / vocal_ride /
                           vocal_tease / drum_trade / stem_swap; without stems from
                           blend / filter / swap / drop_swap / chop.
             cut           tempos too far apart: roll_drop / spinback / brake /
                           washout / echo_out, with B dropped on a strong "1".
             fade          no steady beat at all (ambient, rubato): plain crossfade.
           The same move is never played twice in a row. See src/dj_techniques.py.

  PERFORM  * Both decks are warped (key-locked WSOLA) onto ONE master beat grid whose
             tempo glides from A's BPM to B's across the overlap. Kicks stay locked,
             each track is at its own natural speed whenever it is heard alone, and
             track bodies are always the original, un-stretched audio.
           * Every track is loudness-normalised as a whole (no level jump when a
             transition ends); transitions go through a look-ahead limiter.

The timeline is sample-contiguous by construction, so it can be streamed part by
part (see club_server.py) with no gaps or splices.

References: Zehren et al. 2022; Vande Veire & De Bie 2018; Foote 2000 / Müller FMP;
Verhelst & Roelands 1993 (WSOLA); Krumhansl 1990 (key profiles); ITU-R BS.1770.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import scipy.signal as sig
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from src.dj_techniques import (
    STEM_TECHNIQUES, EQ_TECHNIQUES, CUT_TECHNIQUES, CUT_LEAD_BARS, CUT_TAIL_BARS, CUT_WANTS_DROP,
    ORIGINAL_CUT_BARS, LABELS, DESCRIPTIONS, Timing, as_stereo as _as_stereo, fit_len as _fit_len,
    render_beatmatched, render_cut, stem_features, _stem_parts, hole_db,
    choose_eq_technique, choose_stem_technique, choose_cut_technique,
)
from src import technique_bridge as bridge

_LOG = logging.getLogger("automix.engine")
# Per-render context (set by build_continuous_set): the caller's logger (so engine
# lines carry its session id) and a checkpoint() that may raise to cancel the render.
_CTX_LOG: "contextvars.ContextVar" = contextvars.ContextVar("automix_log", default=_LOG)
_CTX_CHECK: "contextvars.ContextVar" = contextvars.ContextVar("automix_check", default=None)


def _log():
    return _CTX_LOG.get()


def _checkpoint() -> None:
    """Cooperative cancellation point inside long steps (stem separation, warping,
    each technique attempt). The caller's checkpoint raises to abort the render."""
    fn = _CTX_CHECK.get()
    if fn is not None:
        fn()


ANALYSIS_SR = 22050
TARGET_LUFS = -14.0
BEAT_CONF_MIN = 1.45         # beat-synchronous onset contrast: ~1.1-1.4 classical/ambient, 1.5-8+ steady beat
TEMPO_TOLERANCE = 0.18       # widest tempo gap that is still beat-matched (the tempo glides across the blend,
                             # so each deck is only ever stretched by about half of it while it is audible)
TECHNIQUES = bridge.ORIGINAL + STEM_TECHNIQUES + EQ_TECHNIQUES + CUT_TECHNIQUES + ("fade",)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ----------------------------------------------------------------------------- #
# Track analysis
# ----------------------------------------------------------------------------- #
@dataclass
class PhraseGrid:
    sr: int
    tempo: float                 # BPM (scalar, from the smoothed beat grid)
    bar_dur: float               # seconds per 4/4 bar
    downbeats: np.ndarray        # downbeat times (seconds), bar-1 of each bar
    bar_energy: np.ndarray       # RMS per bar
    duration: float              # seconds
    sections: Optional[list] = None  # [{start_sec,end_sec,label:'low'|'high',energy}]
    # --- everything below is new (defaults keep old constructors working) ---
    bt: Optional[np.ndarray] = None       # beat times, extrapolated past both ends
    bk: Optional[np.ndarray] = None       # beat index for each bt (0 = first real beat)
    beat_conf: float = 0.0                # beat-synchronous onset contrast
    bar_db: Optional[np.ndarray] = None   # loudness per bar (dB)
    boundaries: Optional[list] = None     # [(bar_index, novelty_db, energy_change_db)]
    phrase_offset: int = 0                # downbeat index (mod 8) where phrases start
    first_bar: int = 0                    # first audible bar
    usable_end: float = 0.0               # where the track stops being loud enough to mix
    key: Optional[Tuple[int, str]] = None  # (tonic pitch class, 'major'|'minor')
    key_conf: float = 0.0
    camelot: str = ""
    loudness: float = TARGET_LUFS         # integrated LUFS (approx.)
    gain: float = 1.0                     # linear gain that brings the track to target
    fine_env: Optional[np.ndarray] = None  # ~3 ms onset envelope (beat-phase alignment)
    fine_hop: float = 64.0 / ANALYSIS_SR


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=1)


def _phi(g: PhraseGrid, t):
    """Time (s) -> continuous beat position."""
    return np.interp(t, g.bt, g.bk)


def _phi_inv(g: PhraseGrid, k):
    """Beat position -> time (s)."""
    return np.interp(k, g.bk, g.bt)


def _local_period(g: PhraseGrid, t: float) -> float:
    k = float(_phi(g, t))
    return float(_phi_inv(g, k + 4) - _phi_inv(g, k - 4)) / 8.0


def _amp_onset_env(y: np.ndarray, k: int = 64) -> np.ndarray:
    """Rectified log-amplitude rise at k-sample resolution. Only meaningful on a
    narrow-band signal (e.g. the low-passed kick), where amplitude IS the event."""
    n = (len(y) // k) * k
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    fr = np.abs(y[:n]).reshape(-1, k).mean(axis=1)
    lg = np.log1p(200.0 * fr)
    return np.maximum(0.0, np.diff(lg, prepend=lg[0])).astype(np.float32)


def _fine_onset_env(y: np.ndarray, sr: int = ANALYSIS_SR, k: int = 64, n_fft: int = 512) -> np.ndarray:
    """High-resolution (k samples ≈ 3 ms) onset envelope: log-mel spectral flux.
    Unlike a plain amplitude envelope it still sees drum hits inside a loud,
    heavily-compressed master. Frame j is centred on sample j*k."""
    if len(y) < n_fft:
        return np.zeros(max(1, len(y) // k), dtype=np.float32)
    ypad = np.pad(y.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(ypad, n_fft)[::k]
    win = sig.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=32).astype(np.float32)
    env = np.empty(len(frames), dtype=np.float32)
    prev = None
    for a in range(0, len(frames), 8192):
        mag = np.abs(np.fft.rfft(frames[a:a + 8192] * win, axis=1)).astype(np.float32)
        lm = np.log1p(50.0 * (mag @ fb.T))
        first = lm[:1] if prev is None else prev
        d = np.diff(np.vstack([first, lm]), axis=0)
        env[a:a + len(lm)] = np.maximum(d, 0.0).sum(axis=1)
        prev = lm[-1:]
    return env


def _beat_phase(env: np.ndarray, hop: float, beats: np.ndarray, span: float = 0.07
                ) -> Tuple[float, float]:
    """Average the onset envelope around every beat. Returns (offset_sec, contrast):
    where the real attack sits relative to the grid, and how sharply the grid
    'explains' the onsets (≈1 for beatless audio, ≫1 for a steady kick)."""
    w = int(round(span / hop))
    idx = np.round(np.asarray(beats) / hop).astype(int)
    idx = idx[(idx - w >= 0) & (idx + w < len(env))]
    if len(idx) < 8:
        return 0.0, 0.0
    prof = np.zeros(2 * w + 1)
    for o in range(-w, w + 1):
        prof[o + w] = float(np.mean(env[idx + o]))
    prof = np.convolve(prof, [0.25, 0.5, 0.25], mode="same")
    base = float(np.mean(env)) + 1e-9
    i = int(np.argmax(prof))
    frac = 0.0
    if 0 < i < 2 * w:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-12:
            frac = 0.5 * (a - c) / d
    return float((i - w + frac) * hop), float(prof[i] / base)


def _fold(env: np.ndarray, hop: float, period: float, nbins: int) -> np.ndarray:
    """Fold an onset envelope at `period`: mean onset strength per beat-phase bin.
    The right period stacks every beat's attack into one sharp peak."""
    idx = np.minimum((((np.arange(len(env)) * hop) % period) / period * nbins).astype(int), nbins - 1)
    s = np.bincount(idx, weights=env, minlength=nbins)
    c = np.bincount(idx, minlength=nbins)
    return s / np.maximum(c, 1)


def _refine_period(env: np.ndarray, hop: float, p0: float,
                   levels=(30.0, 120.0, None)) -> float:
    """Sharpen a rough beat period to ~0.001% by maximising the folded-envelope peak,
    coarse-to-fine over longer and longer excerpts (a 0.1% tempo error is already a
    30 ms flam after 16 bars — the difference between locked and trainwrecking)."""
    n = len(env)
    best = p0
    span = 0.03 * p0
    cs = np.concatenate([[0.0], np.cumsum(env, dtype=np.float64)])
    for win_sec in levels:
        wl = n if win_sec is None else min(n, int(win_sec / hop))
        if wl < n:                               # the busiest excerpt (drums present)
            starts = np.arange(0, n - wl + 1, max(1, wl // 8))
            a = int(starts[int(np.argmax(cs[starts + wl] - cs[starts]))])
        else:
            a = 0
        seg = env[a:a + wl]
        nbeats = max(8.0, wl * hop / best)
        width = 0.006 / nbeats                   # period error that blurs the peak
        cands = best + np.arange(-span, span + width / 3, width / 3)
        cands = cands[cands > 0.5 * p0]
        sc = [float(np.convolve(_fold(seg, hop, c, 64), [0.25, 0.5, 0.25], mode="same").max())
              for c in cands]
        best = float(cands[int(np.argmax(sc))])
        span = 2.0 * width
        if wl >= n:
            break
    return best


def _beat_grid(y: np.ndarray, asr: int, dp_beats: np.ndarray, fine: np.ndarray,
               fine_hop: float, duration: float) -> Tuple[np.ndarray, float]:
    """Turn the beat tracker's rough beats into a grid a DJ could mix on.

    Dance/pop masters have a machine-steady tempo, so the best grid is a straight
    one: precise period (comb fold), phase taken from the kick (low band) so we
    never lock to the off-beat hat, plus a slow correction for small drift. If that
    explains the onsets worse than the tracker's own beats (live drummer, tempo
    changes), keep the tracker's beats, smoothed and re-centred on the attacks.
    """
    # -- fallback / reference: the tracker's beats, smoothed --
    dp = np.asarray(dp_beats, dtype=float)
    win = min(33, len(dp) - (1 - len(dp) % 2))
    if win >= 5:
        dp = sig.savgol_filter(dp, window_length=win, polyorder=1, mode="interp")
    off, conf_dp = _beat_phase(fine, fine_hop, dp)
    if conf_dp >= 1.3 and abs(off) <= 0.06:
        dp = dp + off
        _, conf_dp = _beat_phase(fine, fine_hop, dp)

    # -- straight grid --
    p0 = float(np.median(np.diff(dp)))
    period = _refine_period(fine, fine_hop, p0)

    # The tracker sometimes locks onto a 3:2 or 4:3 relative of the real tempo (e.g. 89
    # for 134 BPM). Folding at such a period spreads the beats over several phases, so
    # the true period shows a clearly sharper fold — check the neighbours and switch.
    def sharp(pp: float) -> float:
        hh = np.convolve(_fold(fine, fine_hop, pp, 64), [0.25, 0.5, 0.25], mode="same")
        return float(hh.max() / (hh.mean() + 1e-12))
    best_c = sharp(period)
    for mult in (2.0 / 3.0, 3.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0):
        q = _refine_period(fine, fine_hop, period * mult, levels=(30.0,))
        if 0.25 <= q <= 1.2 and sharp(q) > 1.10 * best_c:
            period, best_c = _refine_period(fine, fine_hop, q), sharp(q)
    nb = max(16, int(round(period / fine_hop)))
    h = np.convolve(_fold(fine, fine_hop, period, nb), [0.25, 0.5, 0.25], mode="same")
    sos = sig.butter(4, 150.0 / (asr / 2.0), btype="low", output="sos")
    low = _amp_onset_env(sig.sosfilt(sos, y).astype(np.float32))
    hl = np.convolve(_fold(low, fine_hop, period, nb), [0.25, 0.5, 0.25], mode="same")
    i = int(np.argmax(h))
    if hl.max() >= 1.5 * (hl.mean() + 1e-12):    # a real kick: trust it for on/off-beat
        j = int(np.argmax(hl))
        reach = max(1, int(round(0.04 / (period / nb))))
        cand = (j + np.arange(-reach, reach + 1)) % nb
        i = int(cand[int(np.argmax(h[cand]))])
    phase = (i + 0.5) * period / nb
    while phase - period >= -0.03:                # keep a beat sitting right at t=0
        phase -= period
    grid = phase + period * np.arange(int(np.floor((duration - phase) / period)) + 1)
    grid[0] = max(grid[0], 0.0)
    if len(grid) >= 16:
        # slow drift correction, measured per 32-beat window
        ks, offs = [], []
        for a in range(0, len(grid) - 16, 16):
            o, c = _beat_phase(fine, fine_hop, grid[a:a + 32], span=0.05)
            if c >= 1.5:
                ks.append(a + 16); offs.append(o)
        if len(ks) >= 3:
            offs = sig.medfilt(np.asarray(offs), 3)
            if np.max(np.abs(offs)) > 0.002:
                grid = grid + np.interp(np.arange(len(grid)), ks, offs)
        elif len(ks) >= 1:
            grid = grid + float(np.median(offs))
        _, conf = _beat_phase(fine, fine_hop, grid)
        if conf >= 0.9 * conf_dp:
            return grid, conf
    return dp, conf_dp


def _k_weighted_lufs(y: np.ndarray, sr: int) -> float:
    """Approximate BS.1770 integrated loudness of a mono signal (gated)."""
    def biquad(b, a):
        return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    # high shelf
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    A = 10 ** (G / 40.0); w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    shelf = biquad([A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al),
                    -2 * A * ((A - 1) + (A + 1) * c),
                    A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)],
                   [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al,
                    2 * ((A - 1) - (A + 1) * c),
                    (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al])
    Q, fc = 0.5003270373253953, 38.13547087613982
    w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    hp = biquad([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al])
    z = sig.sosfilt(np.vstack([shelf, hp]), y.astype(np.float64))
    blk = int(0.4 * sr); hop = int(0.1 * sr)
    if len(z) < blk:
        ms = np.array([np.mean(z ** 2) + 1e-12])
    else:
        cs = np.concatenate([[0.0], np.cumsum(z ** 2)])
        starts = np.arange(0, len(z) - blk + 1, hop)
        ms = (cs[starts + blk] - cs[starts]) / blk + 1e-12
    lk = -0.691 + 10 * np.log10(ms)
    keep = ms[lk > -70.0]
    if len(keep) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(keep)) - 10.0
    keep2 = ms[(lk > -70.0) & (lk > rel)]
    if len(keep2) == 0:
        keep2 = keep
    # +3 dB: a centred mono downmix carries the same signal on both stereo channels
    return float(-0.691 + 10 * np.log10(np.mean(keep2)) + 3.0)


def _detect_key(chroma: np.ndarray) -> Tuple[Optional[Tuple[int, str]], float, str]:
    prof = chroma.mean(axis=1)
    if not np.isfinite(prof).all() or prof.std() < 1e-9:
        return None, 0.0, ""
    scores = []
    for tonic in range(12):
        scores.append((float(np.corrcoef(prof, np.roll(_KS_MAJOR, tonic))[0, 1]), tonic, "major"))
        scores.append((float(np.corrcoef(prof, np.roll(_KS_MINOR, tonic))[0, 1]), tonic, "minor"))
    scores.sort(reverse=True)
    best, tonic, mode = scores[0]
    cam = _camelot(tonic, mode)
    # confidence = margin over the best key that is NOT a mixing-compatible neighbour
    rival = next((s for s, t, m in scores[1:] if not _keys_compatible_cam(cam, _camelot(t, m))), best)
    return (tonic, mode), float(best - rival), cam


def _camelot(tonic: int, mode: str) -> str:
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    num = ((major_tonic * 7) % 12 + 7) % 12 + 1
    return f"{num}{'B' if mode == 'major' else 'A'}"


def _keys_compatible_cam(a: str, b: str) -> bool:
    if not a or not b:
        return True
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    if na == nb:
        return True                       # same key or relative major/minor
    d = min((na - nb) % 12, (nb - na) % 12)
    return la == lb and d == 1            # one step round the wheel


def keys_compatible(ga: PhraseGrid, gb: PhraseGrid, min_conf: float = 0.04) -> bool:
    """Harmonic-mixing rule (Camelot wheel). Unknown / unsure keys never block a blend."""
    if ga.key is None or gb.key is None or ga.key_conf < min_conf or gb.key_conf < min_conf:
        return True
    return _keys_compatible_cam(ga.camelot, gb.camelot)


def key_name(g: PhraseGrid) -> str:
    if g.key is None:
        return "?"
    return f"{_NOTE_NAMES[g.key[0]]}{'m' if g.key[1] == 'minor' else ''}"


_GRID_CACHE: "OrderedDict[str, PhraseGrid]" = OrderedDict()
_GRID_CACHE_CAP = 64


def build_phrase_grid(y: np.ndarray, sr: int, hop_length: int = 512,
                      phrase_bars: int = 8) -> PhraseGrid:
    """Full per-track analysis (see module docstring). Cached by audio content."""
    mono = _to_mono(y).astype(np.float32)
    key = hashlib.md5(np.ascontiguousarray(mono).tobytes()).hexdigest() + f"|{sr}|{phrase_bars}"
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        _GRID_CACHE.move_to_end(key)
        return cached
    grid = _analyze(mono, sr, phrase_bars)
    _GRID_CACHE[key] = grid
    while len(_GRID_CACHE) > _GRID_CACHE_CAP:
        _GRID_CACHE.popitem(last=False)
    return grid


def _extend_beats(beats: np.ndarray, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extrapolate the beat grid past both ends so beat<->time maps never clamp."""
    d = np.diff(beats)
    p0 = float(np.median(d[:8])); p1 = float(np.median(d[-8:]))
    n_pre = int(np.ceil(max(0.0, beats[0]) / p0)) + 16
    n_post = int(np.ceil(max(0.0, duration - beats[-1]) / p1)) + 256
    pre = beats[0] - p0 * np.arange(n_pre, 0, -1)
    post = beats[-1] + p1 * np.arange(1, n_post + 1)
    bt = np.concatenate([pre, beats, post])
    bk = np.arange(-n_pre, len(beats) + n_post, dtype=float)
    return bt, bk


def _analyze(mono: np.ndarray, sr: int, phrase_bars: int) -> PhraseGrid:
    duration = len(mono) / sr
    asr = ANALYSIS_SR
    if sr == 2 * asr:
        y = sig.resample_poly(mono, 1, 2).astype(np.float32)
    elif sr == asr:
        y = mono
    else:
        y = librosa.resample(mono, orig_sr=sr, target_sr=asr).astype(np.float32)

    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    loud = _k_weighted_lufs(y, asr) if len(y) > asr // 2 else -70.0
    gain_db = float(np.clip(TARGET_LUFS - loud, -12.0, 9.0)) if loud > -60 else 0.0
    gain = 10 ** (gain_db / 20.0)
    if peak * gain > 0.95:                       # never normalise a track into clipping
        gain = 0.95 / peak
    fine = _fine_onset_env(y)
    fine_hop = 64.0 / asr

    def degenerate() -> PhraseGrid:
        bar = 2.0
        beats = np.arange(0.0, max(duration, 4.0), 0.5)
        bt, bk = _extend_beats(beats, duration)
        dbs = beats[::4]
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9) if len(mono) else 1e-9
        return PhraseGrid(sr, 120.0, bar, dbs, np.full(len(dbs), rms), duration,
                          [{"start_sec": 0.0, "end_sec": duration, "label": "high", "energy": rms}],
                          bt=bt, bk=bk, beat_conf=0.0, bar_db=np.full(len(dbs), 20 * np.log10(rms)),
                          boundaries=[], usable_end=duration, loudness=loud, gain=gain,
                          fine_env=fine, fine_hop=fine_hop)

    if duration < 8.0:
        return degenerate()

    hop = 256
    mel = librosa.feature.melspectrogram(y=y, sr=asr, n_fft=1024, hop_length=hop, n_mels=40)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 70.0)
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=asr, hop_length=hop)
    tempo, bfr = librosa.beat.beat_track(onset_envelope=onset_env, sr=asr, hop_length=hop,
                                         units="frames", trim=False)
    beats = librosa.frames_to_time(bfr, sr=asr, hop_length=hop).astype(float)
    if len(beats) < 16:
        return degenerate()

    beats, conf = _beat_grid(y, asr, beats, fine, fine_hop, duration)
    period = float(np.median(np.diff(beats)))
    tempo = 60.0 / period
    bar_dur = 4.0 * period
    bt, bk = _extend_beats(beats, duration)

    # --- downbeat phase: where do chords change, sections start, and lows hit? ---
    nb = len(beats)
    bf = np.clip(librosa.time_to_frames(beats, sr=asr, hop_length=hop), 0, mel_db.shape[1] - 1)
    bmel = librosa.util.sync(mel_db, bf, aggregate=np.mean, pad=False)      # [40, nb-1]
    chroma = librosa.feature.chroma_stft(y=y, sr=asr, n_fft=4096, hop_length=1024)
    cf = np.clip(librosa.time_to_frames(beats, sr=asr, hop_length=1024), 0, chroma.shape[1] - 1)
    bchr = librosa.util.sync(chroma, cf, aggregate=np.mean, pad=False)
    nbm = min(bmel.shape[1], bchr.shape[1], nb)
    sec_nov = np.zeros(nbm); chr_nov = np.zeros(nbm)
    for k in range(1, nbm):
        a, b = bchr[:, k], bchr[:, k - 1]
        chr_nov[k] = 1.0 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if 4 <= k <= nbm - 4:
            sec_nov[k] = float(np.mean(np.abs(bmel[:, k:k + 4].mean(axis=1) - bmel[:, k - 4:k].mean(axis=1))))
    low_on = librosa.onset.onset_strength(S=mel_db[:4], sr=asr, hop_length=hop)[bf[:nbm]]

    def z(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)
    score = (z([chr_nov[o::4].mean() for o in range(4)])
             + z([np.sort(sec_nov[o::4])[-max(1, len(sec_nov[o::4]) // 8):].mean() for o in range(4)])
             + 0.5 * z([low_on[o::4].mean() for o in range(4)]))
    offset = int(np.argmax(score))
    downbeats = np.asarray(beats[offset::4], dtype=float)
    if len(downbeats) < 2:
        return degenerate()

    # --- per-bar features ---
    nbar = len(downbeats)
    edges = np.append(downbeats, min(duration, downbeats[-1] + bar_dur))
    bar_energy = np.empty(nbar)
    for i in range(nbar):
        seg = mono[int(edges[i] * sr):int(edges[i + 1] * sr)]
        bar_energy[i] = float(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else 1e-9
    bar_db = 20.0 * np.log10(bar_energy)
    dbf = np.clip(librosa.time_to_frames(downbeats, sr=asr, hop_length=hop), 0, mel_db.shape[1] - 1)
    bar_mel = librosa.util.sync(mel_db, dbf, aggregate=np.mean, pad=False)
    if bar_mel.shape[1] < nbar:
        bar_mel = np.pad(bar_mel, ((0, 0), (0, nbar - bar_mel.shape[1])), mode="edge")
    ref_db = float(np.percentile(bar_db, 85))
    audible = np.where(bar_db >= ref_db - 30.0)[0]
    first_bar = int(audible[0]) if len(audible) else 0
    usable = np.where(bar_db >= ref_db - 12.0)[0]
    usable_end = float(edges[int(usable[-1]) + 1]) if len(usable) else duration

    # --- section boundaries: bar-level novelty (4 bars before vs 4 bars after) ---
    nov = np.zeros(nbar); d_e = np.zeros(nbar)
    for i in range(1, nbar):
        lo = max(0, i - 4); hi = min(nbar, i + 4)
        nov[i] = float(np.mean(np.abs(bar_mel[:, i:hi].mean(axis=1) - bar_mel[:, lo:i].mean(axis=1))))
        d_e[i] = float(bar_db[i:hi].mean() - bar_db[lo:i].mean())
    boundaries = []
    for i in range(2, nbar - 1):
        if nov[i] >= 2.5 and nov[i] >= nov[max(0, i - 2):i + 3].max() - 1e-9:
            boundaries.append((i, float(nov[i]), float(d_e[i])))
    res = np.zeros(phrase_bars)
    for i, n_, _ in boundaries:
        res[i % phrase_bars] += n_
    phrase_offset = int(np.argmax(res)) if res.max() > 0 else first_bar % phrase_bars
    # Sections start on phrase lines: pull near-misses (a riser one bar early) onto the grid.
    snapped = {}
    for i, n_, de in boundaries:
        r = (i - phrase_offset) % 4
        j = i - 1 if r == 1 else (i + 1 if r == 3 else i)
        if 1 <= j < nbar and (j not in snapped or snapped[j][1] < n_):
            snapped[j] = (j, n_, float(d_e[j]) if abs(d_e[j]) > abs(de) else de)
    boundaries = [snapped[j] for j in sorted(snapped)]

    cuts = [0] + [i for i, _, _ in boundaries] + [nbar]
    sections = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        en = float(np.mean(bar_energy[a:b]))
        sections.append({"start_sec": float(edges[a]) if a else 0.0,
                         "end_sec": float(edges[b]) if b < nbar else duration,
                         "label": "high" if 20 * np.log10(en) >= ref_db - 4.0 else "low",
                         "energy": en})

    key, key_conf, cam = _detect_key(chroma)
    return PhraseGrid(sr, tempo, bar_dur, downbeats, bar_energy, duration, sections,
                      bt=bt, bk=bk, beat_conf=conf, bar_db=bar_db, boundaries=boundaries,
                      phrase_offset=phrase_offset, first_bar=first_bar, usable_end=usable_end,
                      key=key, key_conf=key_conf, camelot=cam, loudness=loud, gain=gain,
                      fine_env=fine, fine_hop=fine_hop)


# ----------------------------------------------------------------------------- #
# Transition planning
# ----------------------------------------------------------------------------- #
@dataclass
class TransitionPlan:
    blend_bars: int              # pre_bars + post_bars (0 for cuts / fades)
    bar_dur: float               # seconds per bar of A around the switch
    a_switch_sec: float          # t2 in A coordinates (a downbeat)
    a_t1_sec: float              # where the transition starts replacing A's audio
    a_t3_sec: float              # where A is gone
    b_in_sec: float              # B's cue: where B starts sounding from
    transition_type: str         # one of TECHNIQUES (stems: finalised at render time)
    overlap_samples: int         # nominal length (beat-matched overlaps follow the tempo curve)
    family: str = "beatmatch"    # beatmatch | cut | fade
    pre_bars: int = 0            # bars before the switch (B arriving under A)
    post_bars: int = 0           # bars after the switch (A leaving under B)
    switch_offset: int = 0       # samples from the transition start to the switch
    b_switch_sec: float = 0.0    # the point in B that lands on the switch
    metric: float = 1.0          # B beats per A beat (1, 2 or 0.5 for half/double time)
    rate: float = 1.0            # B's beat length relative to A's (0.95 = B is 5% faster)
    tail_samples: int = 0        # cuts: how long the effect rings on over B
    key_compatible: bool = True
    b_has_drop: bool = False     # B's cue point is a real "energy up" moment
    use_stems: bool = False
    features: Optional[Dict] = None
    harmonic_fit: Optional[float] = None
    tempo_a: float = 0.0
    tempo_b: float = 0.0
    key_a: str = "?"
    key_b: str = "?"
    reason: str = ""


def _pow2_floor(x: float, cap: int) -> int:
    """Largest of 32/16/8/4/2/1 bars that fits in x (and under cap); 0 if less than a bar."""
    for p in (32, 16, 8, 4, 2, 1):
        if p <= cap and p <= x + 1e-6:
            return p
    return 0


def _tempo_relation(period_a: float, period_b: float) -> Tuple[float, float]:
    """(metric, rate): how many B beats per A beat (1, 2, 0.5) and how B's beat length
    then compares to A's. Picks the relation with the smallest tempo gap."""
    best = min((1.0, 2.0, 0.5), key=lambda m: abs(np.log(m * period_b / period_a)))
    return best, best * period_b / period_a


def _bar_time(g: PhraseGrid, i: int) -> float:
    return float(g.downbeats[int(np.clip(i, 0, len(g.downbeats) - 1))])


def plan_transition(grid_a: PhraseGrid, grid_b: PhraseGrid,
                    blend_bars: int = 16,
                    recent_types: Optional[list] = None,
                    min_start_sec: float = 0.0,
                    tempo_tolerance: float = TEMPO_TOLERANCE,
                    use_stems: bool = False,
                    play_window: Optional[Tuple[float, float]] = None) -> TransitionPlan:
    """Decide how, where and for how long to mix A into B.

    `min_start_sec` is the earliest point in A that is still unplayed (A's own
    mix-in may have consumed its beginning) — the transition is never placed before it.
    `play_window=(lo, hi)` is quick-mix mode: A plays on its own for lo..hi bars after
    min_start_sec and is then mixed out at the best phrase boundary in that window,
    instead of waiting for its outro (a short DJ set — never a cut inside a song).
    With `use_stems`, beat-matched plans carry a provisional technique; the final one
    is chosen at render time, once the stems show what is actually playing.
    """
    recent = list(recent_types or [])
    half = max(1, int(blend_bars) // 2)
    ga, gb = grid_a, grid_b
    end_a = min(ga.duration, max(ga.usable_end, min_start_sec + 1.0))
    key_ok = keys_compatible(ga, gb)
    common = dict(tempo_a=ga.tempo, tempo_b=gb.tempo, key_a=key_name(ga), key_b=key_name(gb),
                  key_compatible=key_ok)

    b0_bar = gb.first_bar
    b0 = _bar_time(gb, b0_bar) if len(gb.downbeats) else 0.0

    def fade_plan(reason: str) -> TransitionPlan:
        length = float(np.clip(min(8.0, 0.5 * (end_a - min_start_sec), 0.4 * gb.duration), 0.05, 8.0))
        a1 = max(min_start_sec, end_a - length)
        b_in = max(0.0, min(b0, gb.duration - length)) if gb.beat_conf >= BEAT_CONF_MIN else 0.0
        n = int(round(length * ga.sr))
        return TransitionPlan(0, ga.bar_dur, a1 + length / 2, a1, a1 + length, b_in, "fade", n,
                              family="fade", switch_offset=n // 2, b_switch_sec=b_in + length / 2,
                              reason=reason, **common)

    beat_ok = (ga.beat_conf >= BEAT_CONF_MIN and gb.beat_conf >= BEAT_CONF_MIN
               and ga.bt is not None and gb.bt is not None)
    if not beat_ok:
        return fade_plan("no steady beat to lock onto")

    pa = _local_period(ga, max(min_start_sec, end_a - 8 * ga.bar_dur))
    pb = _local_period(gb, b0 + 4 * gb.bar_dur)
    metric, rate = _tempo_relation(pa, pb)
    bar_a = 4.0 * pa
    nbar_a = len(ga.downbeats)
    bset = {i: (n_, de) for i, n_, de in (ga.boundaries or [])}

    def score_out(c: int, post: int, want_post: int, cut: bool) -> float:
        """How good is A's bar c as the switch point?"""
        t = _bar_time(ga, c)
        remaining = max(0.0, (end_a - t) / bar_a)
        s = 0.0 if play_window else -(remaining - post) / 8.0   # full sets: don't throw away A's ending
        if c in bset:
            s += (2.0 if cut else 1.5) if bset[c][1] < -1.0 else 0.75   # outro start
        rel = (c - ga.phrase_offset) % 8
        s += 0.5 if rel == 0 else (0.25 if rel == 4 else 0.0)
        if want_post:
            s -= (want_post - post) / want_post
        if t < 0.5 * ga.duration and not play_window:
            s -= 3.0                                        # far too early to leave
        return s

    def in_window(a_start: float) -> bool:
        """Quick-mix mode: does A's solo stretch (min_start -> a_start) fit the window?"""
        if not play_window:
            return True
        solo = (a_start - min_start_sec) / bar_a
        return play_window[0] - 0.05 <= solo <= play_window[1] + 0.05

    # B's first real change (end of its intro) — the natural thing to land on the switch.
    # It must be a point where B actually has weight: handing over to a track that is
    # still in a near-silent intro leaves dead air however the handover is done.
    ref_b = float(np.percentile(gb.bar_db, 85)) if gb.bar_db is not None and len(gb.bar_db) else 0.0

    def b_loud_from(bar: int, n: int = 4) -> bool:
        seg = gb.bar_db[bar:bar + n] if gb.bar_db is not None else []
        return len(seg) > 0 and float(np.mean(seg)) >= ref_b - 8.0

    intro_end, b_has_drop = None, False
    cands = [(i, de) for i, _, de in (gb.boundaries or []) if 2 <= i - b0_bar <= 64]
    loud = [(i, de) for i, de in cands if b_loud_from(i)]
    # Land B's first strong section on the switch. In quick-mix mode, if B's intro is too
    # short for a full blend, cue B in from further along (a DJ cue point) so the blend
    # keeps its length. Full sets play every song from its start, so they never skip ahead.
    want_pre = half / metric
    long_enough = ([(i, de) for i, de in loud if want_pre <= i - b0_bar <= want_pre + 32]
                   if play_window else [])
    pick = (long_enough[0] if long_enough else
            next(iter(loud), cands[0] if cands else None))
    if pick is not None:
        intro_end, b_has_drop = pick[0], pick[1] > 1.0
    elif not b_loud_from(b0_bar):                     # no boundary found, but B starts quiet
        first_loud = next((i for i in range(b0_bar + 2, min(len(gb.downbeats), b0_bar + 65))
                           if b_loud_from(i)), None)
        if first_loud is not None:
            intro_end, b_has_drop = first_loud, True

    # ---- tempos too far apart to beat-match: make the change an event ----
    if abs(np.log(rate)) > np.log(1.0 + tempo_tolerance):
        best = None
        for c in range(nbar_a):
            t = _bar_time(ga, c)
            if t < min_start_sec + 4 * bar_a or t > end_a - (ORIGINAL_CUT_BARS + 0.25) * bar_a:
                continue                                    # A must have 2 more bars to work with
            if not in_window(t):
                continue
            sc = score_out(c, 0, 0, True)
            if best is None or sc > best[0]:
                best = (sc, c)
        if best is None:
            return fade_plan("tempos too far apart and no room for a cut")
        c = best[1]
        sw = _bar_time(ga, c)
        ref = float(np.percentile(ga.bar_db, 85)) if ga.bar_db is not None and len(ga.bar_db) else 0.0
        hot = bool(ga.bar_db is not None and len(ga.bar_db)
                   and float(np.mean(ga.bar_db[max(0, c - 4):max(1, c)])) >= ref - 4.0)
        # Where could B come in with weight? Its first "energy up" boundary, else its start.
        drop = next((i for i, _, de in (gb.boundaries or [])
                     if de > 1.0 and 2 <= i - b0_bar <= 48 and b_loud_from(i, 2)
                     and gb.duration - _bar_time(gb, i) > 16 * gb.bar_dur), None)
        if drop is None and intro_end is not None and b_loud_from(intro_end, 2):
            drop = intro_end
        hot_cue = drop if drop is not None else (b0_bar if b_loud_from(b0_bar, 2) else None)
        technique = choose_cut_technique(hot, (sw - min_start_sec) / bar_a, recent,
                                         b_cue_hot=hot_cue is not None)
        b_cue = _bar_time(gb, hot_cue) if (technique in CUT_WANTS_DROP and hot_cue is not None) else b0
        lead_bars = CUT_LEAD_BARS.get(technique, 0.0)        # original techniques start AT the switch
        tail_bars = CUT_TAIL_BARS.get(technique, ORIGINAL_CUT_BARS)
        a_t1 = float(_phi_inv(ga, float(_phi(ga, sw)) - 4.0 * lead_bars))
        tail = int(round(tail_bars * bar_a * ga.sr))
        n = int(round((sw - a_t1) * ga.sr)) + tail
        gap = (max(rate, 1 / rate) - 1) * 100
        return TransitionPlan(0, bar_a, sw, a_t1, sw, b_cue, technique, n, family="cut",
                              switch_offset=int(round((sw - a_t1) * ga.sr)), b_switch_sec=b_cue,
                              metric=metric, rate=rate, tail_samples=tail,
                              b_has_drop=b_cue > b0 + 1e-6,
                              reason=f"tempos are {gap:.0f}% apart — too far to beat-match",
                              **common)

    # ---- beat-matched ----
    # With stems a key clash is handled per stem, so the blend can stay long.
    want = half if (key_ok or use_stems) else min(half, 4)
    if intro_end is not None:
        pre_cap_b = _pow2_floor((intro_end - b0_bar) / metric, want)
    else:
        pre_cap_b = min(want, 8)
    b_room = (gb.duration - b0) / (bar_a * rate)            # A-bars of B available

    best = None
    for pre in (32, 16, 8, 4, 2, 1):
        if pre > pre_cap_b:
            continue
        for c in range(nbar_a):
            t = _bar_time(ga, c)
            k = float(_phi(ga, t))
            if float(_phi_inv(ga, k - 4 * pre)) < min_start_sec + 0.05:
                continue
            if not in_window(float(_phi_inv(ga, k - 4 * pre))):
                continue
            remaining = (end_a - t) / bar_a
            post = _pow2_floor(min(remaining + 0.25, b_room - pre), want)
            if post < 1:
                continue
            sc = score_out(c, post, want, False) - 1.2 * np.log2(want / pre)
            if best is None or sc > best[0]:
                best = (sc, c, pre, post)
    if best is None:
        return fade_plan("not enough room for a beat-matched blend")
    _, c, pre, post = best

    technique = ("stem_blend" if key_ok else "stem_swap") if use_stems else \
        choose_eq_technique(key_ok, pre, b_has_drop, recent)

    sw = _bar_time(ga, c)
    k_sw = float(_phi(ga, sw))
    a_t1 = float(_phi_inv(ga, k_sw - 4 * pre))
    a_t3 = float(_phi_inv(ga, k_sw + 4 * post))
    if intro_end is not None:
        b_sw = _bar_time(gb, intro_end)
        b_in = float(_phi_inv(gb, float(_phi(gb, b_sw)) - 4 * pre * metric))
    else:
        b_in = b0
        b_sw = float(_phi_inv(gb, float(_phi(gb, b0)) + 4 * pre * metric))
    n = int(round((a_t3 - a_t1) * ga.sr))
    gap = (max(rate, 1 / rate) - 1) * 100
    why = "keys are compatible" if key_ok else f"keys clash ({ga.camelot} vs {gb.camelot})"
    if gap >= 1.0:
        why += f"; tempo glides {gap:.0f}% across the blend"
    if metric != 1.0:
        why += "; half/double-time match"
    return TransitionPlan(pre + post, bar_a, sw, a_t1, a_t3, b_in, technique, n,
                          pre_bars=pre, post_bars=post,
                          switch_offset=int(round((sw - a_t1) * ga.sr)), b_switch_sec=b_sw,
                          metric=metric, rate=rate, b_has_drop=b_has_drop,
                          use_stems=use_stems, reason=why, **common)


# ----------------------------------------------------------------------------- #
# DSP helpers
# ----------------------------------------------------------------------------- #
def _equal_power(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    return np.sqrt(0.5 * (1 + np.cos(np.pi * t))), np.sqrt(0.5 * (1 - np.cos(np.pi * t)))


def _limit(x: np.ndarray, sr: int, ceiling: float = 0.95) -> np.ndarray:
    """Transparent look-ahead peak limiter. Gain is exactly 1 wherever the signal is
    already under the ceiling, so limited and untouched parts join seamlessly."""
    if x.size == 0:
        return x
    pk = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    if float(pk.max()) <= ceiling:
        return x
    look = max(8, int(0.005 * sr))
    env = maximum_filter1d(pk, size=2 * look + 1, mode="nearest")
    g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    g = uniform_filter1d(g, size=look, mode="nearest")
    return (x * (g[:, None] if x.ndim == 2 else g)).astype(x.dtype)


def _headroom(y: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (ceiling / m) if m > ceiling else y


def _wsola(x: np.ndarray, pos: np.ndarray, win: int = 2048, tol: int = 440
           ) -> Tuple[np.ndarray, int]:
    """Key-locked time warp along an arbitrary time map (WSOLA).

    `pos[j]` is where in `x` (samples) output frame j — at output sample j*win/2 —
    should come from. Each frame is taken from the best waveform-aligned spot within
    `tol` of that target, so pitch is untouched and timing follows the map to within
    a few ms. Returns (out, next_src): `out` has len(pos)*win/2 samples, and playing
    x[next_src:] straight after it is seamless (it is the natural continuation of
    the last frame) — that is how a warped transition hands back to original audio.
    """
    H = win // 2
    J = len(pos)
    D = 4                                        # search on a 4x decimated mono copy
    pad = win + tol + 8
    lo = max(0, int(np.floor(pos.min())) - pad)
    hi = min(len(x), int(np.ceil(pos.max())) + win + H + pad)
    xr = np.zeros((max(0, hi - lo) + 2 * pad + win, x.shape[1]), dtype=np.float32)
    if hi > lo:
        xr[pad:pad + (hi - lo)] = x[lo:hi]
    base = lo - pad                               # xr index i  <->  x index i + base
    mono = xr.mean(axis=1)
    md = sig.resample_poly(mono, 1, D).astype(np.float32)
    wd = win // D
    w = sig.get_window("hann", win, fftbins=True).astype(np.float32)
    w0 = w.copy(); w0[:H] = 1.0
    out = np.zeros((J * H + win, x.shape[1]), dtype=np.float32)
    keep = tol // 4                               # stay within ~2.5 ms of the map
    nlag = 2 * (tol // D) + 1
    taper = 1.0 - 0.5 * (np.linspace(-1.0, 1.0, nlag) ** 2)   # prefer staying on the map
    max_p = len(xr) - win - 1

    p_prev = 0
    for j in range(J):
        tgt = int(round(pos[j])) - base
        if j == 0:
            p = tgt
        else:
            nat = p_prev + H
            if abs(nat - tgt) <= keep:
                p = nat                           # still on the map: just keep playing
            else:
                tm = md[nat // D:nat // D + wd]
                a = max(0, (tgt - tol) // D)
                region = md[a:a + wd + nlag - 1]
                if len(tm) < wd or len(region) < wd + nlag - 1:
                    p = tgt
                else:
                    c = np.correlate(region, tm, mode="valid")
                    cs = np.concatenate([[0.0], np.cumsum(region.astype(np.float64) ** 2)])
                    en = np.sqrt(cs[wd:wd + nlag] - cs[:nlag]) + 1e-6
                    p = (a + int(np.argmax(c / en * taper))) * D
                    # refine to the sample at full rate
                    ref = mono[nat:nat + H]
                    cand = range(max(0, p - D), min(max_p, p + D) + 1)
                    p = max(cand, key=lambda q: float(np.dot(mono[q:q + H], ref)))
        p = int(np.clip(p, 0, max_p))
        out[j * H:j * H + win] += xr[p:p + win] * (w0 if j == 0 else w)[:, None]
        p_prev = p
    return out[:J * H], int(p_prev + H + base)


def _warp(x: np.ndarray, pos: np.ndarray, H: int = 1024) -> Tuple[np.ndarray, int]:
    """Play x along the sample-position map `pos` (one entry per H output samples)."""
    J = len(pos)
    if np.max(np.abs(pos - (pos[0] + np.arange(J) * H))) < 24:
        p0 = max(0, int(round(pos[0])))             # already at the right speed
        return _fit_len(x[p0:p0 + J * H], J * H), p0 + J * H
    return _wsola(x, pos, win=2 * H)


# ----------------------------------------------------------------------------- #
# Rendering one transition
# ----------------------------------------------------------------------------- #
_SEP = None
import threading as _threading
_SEP_LOCK = _threading.Lock()


def _get_separator():
    """Lazy, shared StemSeparator (cached separation, MPS/CUDA-auto)."""
    global _SEP
    if _SEP is None:
        from src.stem_separator import StemSeparator
        _SEP = StemSeparator()
    return _SEP


def stems_available() -> bool:
    """True when demucs (and torch) can be imported — the stem techniques need it."""
    try:
        import importlib.util
        return (importlib.util.find_spec("demucs") is not None
                and importlib.util.find_spec("torch") is not None)
    except Exception:
        return False


def _separate(seg: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Stems for one overlap segment. Raises if the separator could not really
    separate (its own fallback puts the whole mix in 'other', which would silently
    turn every stem technique into a plain crossfade)."""
    _checkpoint()
    with _SEP_LOCK:                                   # one model, shared across render threads
        st = _get_separator().separate_stems(seg, sr)
    _checkpoint()
    rest = sum(float(np.mean(np.asarray(st[k], dtype=np.float64) ** 2))
               for k in ("drums", "bass", "vocals") if k in st)
    if rest < 1e-10 and float(np.mean(seg.astype(np.float64) ** 2)) > 1e-8:
        raise RuntimeError("separator returned no stems")
    return st


def _local_grid(g: PhraseGrid, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Re-fit a straight beat grid to the onsets actually found around [t0, t1].

    The whole-track grid is right on average; a DJ only cares that it is right HERE,
    for these 30 seconds. If a locally fitted grid explains the onsets clearly better
    (tempo drifted, or the global phase sits between the hits) it is used for the
    transition; otherwise the global grid is kept. Returns (beat_times, beat_indices,
    local onset contrast)."""
    hop, env = g.fine_hop, g.fine_env
    p0 = _local_period(g, 0.5 * (t0 + t1))
    lo, hi = max(0.0, t0 - 8 * p0), min(g.duration, t1 + 8 * p0)
    i0, i1 = int(lo / hop), min(len(env), int(hi / hop))
    seg = env[i0:i1]
    beats_g = g.bt[(g.bt >= lo) & (g.bt <= hi)]
    _, conf_g = _beat_phase(env, hop, beats_g, span=0.05)
    if len(seg) * hop < 24 * p0 or len(beats_g) < 16:
        return g.bt, g.bk, conf_g
    width = 0.006 / (len(seg) * hop / p0)
    cands = p0 + np.arange(-0.015 * p0, 0.015 * p0 + width / 3, width / 3)
    sm = lambda h: np.convolve(np.r_[h[-1:], h, h[:1]], [0.25, 0.5, 0.25], mode="valid")
    P = float(cands[int(np.argmax([sm(_fold(seg, hop, c, 64)).max() for c in cands]))])
    nb = max(16, int(round(P / hop)))
    h = sm(_fold(seg, hop, P, nb))
    t_ref = float(beats_g[len(beats_g) // 2])                 # a global beat mid-region
    gi = int(((t_ref - i0 * hop) % P) / P * nb) % nb
    reach = max(1, int(round(0.12 * nb)))
    near = (gi + np.arange(-reach, reach + 1)) % nb
    i_near = int(near[int(np.argmax(h[near]))])
    i_pk = i_near if h[i_near] >= 0.8 * h.max() else int(np.argmax(h))   # only jump for a clearly better phase
    first = i0 * hop + (i_pk + 0.5) * P / nb
    n_lo = int(np.floor((first - lo) / P)) + 32
    n_hi = int(np.ceil((hi - first) / P)) + 64
    bt = first + P * np.arange(-n_lo, n_hi + 1)
    _, conf_l = _beat_phase(env, hop, bt[(bt >= lo) & (bt <= hi)], span=0.05)
    if conf_l < 1.5 or conf_l < 1.1 * conf_g:
        return g.bt, g.bk, conf_g
    j = int(np.argmin(np.abs(bt - t_ref)))                     # keep the global beat numbering
    bk = float(np.round(_phi(g, bt[j]))) + (np.arange(len(bt)) - j).astype(float)
    return bt, bk, conf_l


def harmonic_fit(seg_a: np.ndarray, seg_b: np.ndarray, sr: int,
                 stems_a: Optional[Dict] = None, stems_b: Optional[Dict] = None) -> float:
    """How well the two decks' notes sit together in this overlap (-1 clash … +1 same key).

    Key LABELS are often unreliable (a track can score ~equal for several keys), and
    an 'unknown' key used to be waved through as compatible — which is how a
    melody got laid over a track a tritone away. This listens to the actual pitch
    content instead: correlation of the two decks' mean chroma (melodic stems only
    when stems are available). Calibrated: A minor/E minor +0.36, A minor/D minor
    +0.53, C minor/F# minor -0.37.
    """
    def mel(seg, stems):
        x = seg
        if stems is not None:
            x = sum(np.asarray(stems[k], dtype=np.float32) for k in ("other", "vocals") if k in stems)
        m = _to_mono(np.asarray(x, dtype=np.float32))
        if sr == 2 * ANALYSIS_SR:
            m = sig.resample_poly(m, 1, 2).astype(np.float32)
        c = librosa.feature.chroma_stft(y=m, sr=ANALYSIS_SR if sr == 2 * ANALYSIS_SR else sr,
                                        n_fft=8192, hop_length=4096)
        v = c.mean(axis=1)
        return v - v.mean()
    try:
        a, b = mel(seg_a, stems_a), mel(seg_b, stems_b)
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d > 1e-9 else 0.0
    except Exception:
        return 0.0


HARMONIC_OK = 0.10           # >= this: melodies may overlap
HARMONIC_CLASH = -0.10       # < this: treat as a clash even if the key labels agree


def _tempo_map(plan: TransitionPlan, ga: PhraseGrid, gb: PhraseGrid):
    """The master clock of a beat-matched overlap.

    Both decks follow ONE beat grid whose tempo glides from A's BPM (at the start,
    where only A is heard) to B's BPM (at the end, where only B is heard). Each deck
    is therefore at its own natural speed when it is alone, and the stretching is
    shared around the switch — no jump in tempo, no deck stuck at the other's BPM.

    Returns (kk, T, ta, tb, beats_a, beats_b): master beat positions, the output time
    of each, the A / B source time that must be playing at that beat, and the beat
    times (locally re-fitted where that helps) each deck was aligned by.
    """
    K = 4 * (plan.pre_bars + plan.post_bars)
    kk = np.linspace(0.0, K, 8 * K + 1)
    span_a = K * plan.bar_dur / 4.0
    bta, bka, _ = _local_grid(ga, plan.a_t1_sec, plan.a_t1_sec + span_a)
    btb, bkb, _ = _local_grid(gb, plan.b_in_sec, plan.b_in_sec + span_a * plan.rate)
    ka0 = float(np.interp(plan.a_t1_sec, bta, bka))           # A starts exactly at a_t1
    frac = ka0 - np.round(ka0)                                # ... which may sit just off a local beat
    kb0 = float(np.round(np.interp(plan.b_in_sec, btb, bkb))) + plan.metric * frac
    ta = np.interp(ka0 + kk, bka, bta)
    tb = np.interp(kb0 + plan.metric * kk, bkb, btb)
    pa, pb = np.gradient(ta, kk), np.gradient(tb, kk)        # native seconds per master beat
    x = np.clip((kk / K - 0.08) / 0.84, 0.0, 1.0)
    w = x * x * (3.0 - 2.0 * x)
    P = np.exp((1.0 - w) * np.log(pa) + w * np.log(pb))
    T = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(kk))])
    return kk, T, ta, tb, bta, btb


def _render_transition(cur: np.ndarray, grid_a: PhraseGrid, nxt: np.ndarray, grid_b: PhraseGrid,
                       plan: TransitionPlan, sr: int, use_stems: bool = False,
                       recent: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Render one transition. `cur`/`nxt` are loudness-normalised stereo tracks.

    Returns (overlap, lead, resume_b_sec):
      overlap  the transition audio (replaces A from a_t1 onwards)
      lead     a few ms of B that complete the last warp frame (may be empty)
      resume   where in B (seconds) original, untouched playback continues after that
    May finalise `plan` (technique chosen from the stems, real overlap length).
    """
    recent = list(recent or [])
    a1 = max(0, int(round(plan.a_t1_sec * sr)))
    b1 = max(0, int(round(plan.b_in_sec * sr)))
    empty = np.zeros((0, 2), dtype=np.float32)

    if plan.family == "fade":
        N = plan.overlap_samples
        seg_a, seg_b = _fit_len(cur[a1:a1 + N], N), _fit_len(nxt[b1:b1 + N], N)
        fo, fi = _equal_power(N)
        out = _limit(seg_a * fo[:, None] + seg_b * fi[:, None], sr)
        return out.astype(np.float32), empty, plan.b_in_sec + N / sr

    if plan.family == "cut":
        sw = max(a1, int(round(plan.a_switch_sec * sr)))
        beat = max(1, int(round(plan.bar_dur * sr / 4.0)))
        if plan.transition_type in bridge.ORIGINAL_CUTS:
            # original technique on a short segment starting at the switch: A's last
            # bars (with the effect) against B from its cue
            n = plan.tail_samples
            out = bridge.run_original(plan.transition_type, _fit_len(cur[sw:sw + n], n),
                                      _fit_len(nxt[b1:b1 + n], n), sr, plan.tempo_a, plan.tempo_b,
                                      switch_ratio=0.5, beat=beat)
            if sw > a1:
                out = np.vstack([cur[a1:sw], out])
            out = _limit(out.astype(np.float32), sr)
        else:
            out = _limit(render_cut(plan.transition_type, cur, a1, sw, nxt, b1,
                                    plan.tail_samples, beat, sr), sr)
        plan.overlap_samples = len(out)
        return out.astype(np.float32), empty, plan.b_in_sec + plan.tail_samples / sr

    # ---- beat-matched: both decks follow one gliding master tempo ----
    H = 1024
    kk, T, ta, tb, bta, btb = _tempo_map(plan, grid_a, grid_b)
    N = int(round(T[-1] * sr))
    # "nudge the jog wheel": line the real attacks up, not just the grid estimates
    shift = 0.0
    ba = bta[(bta >= ta[0]) & (bta <= ta[-1])]
    bb = btb[(btb >= tb[0]) & (btb <= tb[-1])]
    da, ca = _beat_phase(grid_a.fine_env, grid_a.fine_hop, ba)
    db, cb = _beat_phase(grid_b.fine_env, grid_b.fine_hop, bb)
    if ca >= 1.5 and cb >= 1.5:
        shift = float(np.clip(db - da, -0.04, 0.04))

    J = int(np.ceil(N / H))
    tau = np.arange(J) * H / sr
    kap = np.interp(tau, T, kk)
    over = np.maximum(0.0, tau - T[-1])                  # past the end: native speed
    src_a = np.interp(kap, kk, ta) + over
    src_b = np.interp(kap, kk, tb) + over + shift
    _checkpoint()
    wa, _ = _warp(cur, src_a * sr, H)
    _checkpoint()
    wb, next_src = _warp(nxt, src_b * sr, H)
    seg_a, seg_b = wa[:N], wb[:N]

    bars = np.arange(plan.pre_bars + plan.post_bars + 1)
    timing = Timing(N, sr, np.interp(4.0 * bars, kk, T) * sr, plan.pre_bars, plan.post_bars)
    technique, parts, raw = plan.transition_type, None, None
    if use_stems:
        try:
            raw = (_separate(seg_a, sr), _separate(seg_b, sr))
            parts = (_stem_parts(seg_a, raw[0], sr), _stem_parts(seg_b, raw[1], sr))
        except Exception as e:
            _log().warning("stems unavailable for this transition (%s); using the full-mix techniques", e)
            parts = raw = None
    # Do the notes actually fit? Listen to the overlap rather than trust key labels.
    fit = harmonic_fit(seg_a, seg_b, sr, raw[0] if raw else None, raw[1] if raw else None)
    plan.harmonic_fit = fit
    key_ok = fit >= HARMONIC_OK or (plan.key_compatible and fit >= HARMONIC_CLASH
                                    and grid_a.key_conf >= 0.15 and grid_b.key_conf >= 0.15)
    if key_ok != plan.key_compatible:
        plan.reason = plan.reason.replace("keys are compatible", "").replace("keys clash", "labelled keys clash")
        plan.reason = (f"notes {'fit' if key_ok else 'clash'} in the overlap (harmonic fit {fit:+.2f})"
                       + ("; " + plan.reason.strip("; ") if plan.reason.strip("; ") else ""))
    plan.key_compatible = key_ok
    if not use_stems and technique in EQ_TECHNIQUES + tuple(bridge.EXECUTOR_FX):
        technique = choose_eq_technique(key_ok, plan.pre_bars, plan.b_has_drop, recent)
    ctx = {"key_ok": key_ok}
    if parts is not None:
        try:
            plan.features = stem_features(parts[0], parts[1], timing)
            technique = choose_stem_technique(plan.features, plan.key_compatible, plan.pre_bars,
                                              plan.post_bars, recent, plan.b_has_drop)
        except Exception as e:
            _log().warning("stem analysis failed (%s); using the full-mix techniques", e)
            parts = raw = None
    if parts is None and (technique in STEM_TECHNIQUES or technique in bridge.CONVERSATIONS
                          or technique in bridge.EXECUTOR_STEM):
        technique = choose_eq_technique(plan.key_compatible, plan.pre_bars, plan.b_has_drop, recent)

    def render(name: str) -> np.ndarray:
        if name in bridge.ORIGINAL:                      # the project's original library
            return bridge.run_original(name, seg_a, seg_b, sr, plan.tempo_a, plan.tempo_b,
                                       stems_a=raw[0] if raw else None,
                                       stems_b=raw[1] if raw else None,
                                       switch_ratio=timing.at(timing.S) / max(1, N),
                                       beat=timing.beat())
        return render_beatmatched(name, seg_a, seg_b, timing, ctx,
                                  parts=parts if name in STEM_TECHNIQUES + ("chop",) and parts else None)
    # Render; if a technique fails or leaves seconds of dead air on THIS material,
    # pick the next best one (a real quality gate: it only rejects audible holes).
    banned: List[str] = []
    overlap = None
    for attempt in range(4):
        _checkpoint()
        try:
            overlap = render(technique)
            hole = hole_db(overlap, seg_a, seg_b, sr)
            if hole >= -7.0 or attempt == 3:
                break
            _log().info("'%s' left a %.0f dB hole on this material; choosing another move", technique, hole)
        except Exception as e:
            _log().warning("technique '%s' failed (%s: %s); choosing another move", technique, type(e).__name__, e)
            overlap = None
        banned.append(technique)
        if attempt == 2:                                  # last resort: the plain, safe move
            nxt_t = ("stem_blend" if plan.key_compatible else "stem_swap") if parts is not None \
                else ("blend" if plan.key_compatible else "swap")
        elif parts is not None:
            nxt_t = choose_stem_technique(plan.features, plan.key_compatible, plan.pre_bars,
                                          plan.post_bars, recent, plan.b_has_drop, exclude=banned)
        else:
            nxt_t = choose_eq_technique(plan.key_compatible, plan.pre_bars, plan.b_has_drop,
                                        recent, exclude=banned)
        if nxt_t in banned:                               # nothing new to try
            if overlap is not None:
                break
            nxt_t = "blend"
        technique = nxt_t
    if overlap is None:
        overlap = render_beatmatched("blend", seg_a, seg_b, timing, ctx)
        technique = "blend"

    plan.transition_type, plan.use_stems = technique, parts is not None
    plan.overlap_samples, plan.a_t3_sec = N, float(ta[-1])
    plan.switch_offset = int(round(timing.at(timing.S)))
    full = _limit(np.vstack([overlap, wb[N:]]).astype(np.float32), sr)
    return full[:N], full[N:], next_src / sr


def _marker_info(plan: TransitionPlan) -> Dict:
    gap = (max(plan.rate, 1.0 / plan.rate) - 1.0) * 100.0 if plan.rate > 0 else 0.0
    name = plan.transition_type
    original = name in bridge.ORIGINAL and not (plan.family == "cut" and name not in bridge.ORIGINAL_CUTS)
    return {"transition_type": name,
            "label": (bridge.LABELS if original else LABELS).get(name, LABELS.get(name, name)),
            "description": (bridge.DESCRIPTIONS if original else DESCRIPTIONS).get(
                name, DESCRIPTIONS.get(name, "")),
            "library": "original" if original else "club",
            "family": plan.family, "stems": bool(plan.use_stems), "bars": plan.blend_bars,
            "bpm_from": round(plan.tempo_a, 1), "bpm_to": round(plan.tempo_b, 1),
            "key_from": plan.key_a, "key_to": plan.key_b,
            "key_compatible": bool(plan.key_compatible),
            "harmonic_fit": None if plan.harmonic_fit is None else round(float(plan.harmonic_fit), 2),
            "tempo_gap_pct": round(gap, 1), "reason": plan.reason}


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #
def render_club_mix(y_a: np.ndarray,
                    y_b: np.ndarray,
                    sr: int = 44100,
                    blend_bars: int = 16,
                    context_bars: int = 8,
                    phrase_bars: int = 8,
                    recent_types: Optional[list] = None,
                    use_stems: Optional[bool] = None) -> Tuple[np.ndarray, Dict]:
    """Render the A→B transition with some lead-in / lead-out context.
    `use_stems=None` uses demucs stems whenever it is installed.
    Returns (stereo_audio, plan_info)."""
    grid_a = build_phrase_grid(y_a, sr, phrase_bars=phrase_bars)
    grid_b = build_phrase_grid(y_b, sr, phrase_bars=phrase_bars)
    ya = _as_stereo(np.asarray(y_a)).astype(np.float32) * np.float32(grid_a.gain)
    yb = _as_stereo(np.asarray(y_b)).astype(np.float32) * np.float32(grid_b.gain)

    if use_stems is None:
        use_stems = stems_available()
    plan = plan_transition(grid_a, grid_b, blend_bars=blend_bars, recent_types=recent_types,
                           use_stems=use_stems)
    overlap, glide, resume = _render_transition(ya, grid_a, yb, grid_b, plan, sr, use_stems,
                                                recent=recent_types)

    ctx = int(round(context_bars * plan.bar_dur * sr))
    a_t1 = int(round(plan.a_t1_sec * sr))
    r = int(round(resume * sr))
    final = np.vstack([ya[max(0, a_t1 - ctx):a_t1], overlap, glide, yb[r:r + ctx]]).astype(np.float32)

    info = {
        'technique': plan.transition_type, 'family': plan.family,
        'what_happens': _marker_info(plan)['description'], 'reason': plan.reason,
        'stems_used': plan.use_stems, 'stem_features': plan.features,
        'tempo_a': grid_a.tempo, 'tempo_b': grid_b.tempo, 'b_beat_length_vs_a': plan.rate,
        'key_a': f"{plan.key_a} ({grid_a.camelot})", 'key_b': f"{plan.key_b} ({grid_b.camelot})",
        'key_compatible': plan.key_compatible,
        'bar_dur_sec': plan.bar_dur, 'blend_bars': plan.blend_bars,
        'pre_bars': plan.pre_bars, 'post_bars': plan.post_bars,
        'transition_type': plan.transition_type,
        't1_sec_in_A': plan.a_t1_sec, 't2_switch_sec_in_A': plan.a_switch_sec,
        't3_sec_in_A': plan.a_t3_sec, 'b_mix_in_sec': plan.b_in_sec,
        'b_switch_sec': plan.b_switch_sec, 'b_resume_native_sec': resume,
        'loudness_a_lufs': grid_a.loudness, 'loudness_b_lufs': grid_b.loudness,
        'output_sec': len(final) / sr,
    }
    return final, info


def _resolve_track(t) -> np.ndarray:
    """A track may be a raw audio array OR a zero-arg callable (lazy loader).

    Lazy loaders let a caller stream tracks off disk one at a time instead of
    holding the whole playlist decoded in RAM — the set engine only ever needs
    the current track and the next one, so at most two are resident at once.
    Returns a stereo float32 array.
    """
    if callable(t):
        t = t()
    return _as_stereo(np.asarray(t)).astype(np.float32)


def build_continuous_set(tracks,
                         sr: int = 44100,
                         blend_bars: int = 16,
                         phrase_bars: int = 8,
                         progress=None,
                         use_stems: Optional[bool] = None,
                         on_part=None,
                         checkpoint=None,
                         on_skip=None,
                         logger=None,
                         play_bars: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, list]:
    """Chain N tracks into ONE continuous, gapless, DJ-mixed timeline:

        [track0 → t1] [0→1 transition] [track1, original audio → t1] [1→2 transition] ...

    `tracks` is a list of audio arrays (mono or stereo) or zero-arg loaders.
    Returns (stereo_audio, markers); markers describe each track-body and
    transition span in seconds (plus technique / BPM / key details for a UI).
    Marker track indexes refer to positions in `tracks`.

    on_part(part_audio, marker_or_None) is called with each contiguous piece as it
    is produced, so a server can stream the set while it renders. When streaming,
    the full set is never accumulated in RAM (the returned audio is empty).

    checkpoint()  called often, also inside long steps (stem separation, warping);
                  raise from it to cancel. Only exceptions that are not `Exception`
                  subclasses (e.g. a BaseException) are guaranteed to propagate.
    on_skip(index, reason)  a track that cannot be loaded/analysed, or whose
                  transition cannot be rendered at all, is skipped (the set carries
                  on with the next one) and reported here.
    logger        logging.Logger / LoggerAdapter for engine messages.
    play_bars     quick-mix mode, e.g. (4, 8): after each track has been mixed in it
                  plays on its own for that many bars, then the next transition starts
                  at the best phrase boundary in that window. Every song still plays
                  continuously — tracks are mixed out early, never cut and re-joined.
                  The first track gets a longer window (it has no mix-in before it).
    """
    _emit = on_part or (lambda p, m=None: None)
    accumulate = on_part is None
    if use_stems is None:                 # stems whenever demucs is installed
        use_stems = stems_available()
    tok_log = _CTX_LOG.set(logger or _LOG)
    tok_chk = _CTX_CHECK.set(checkpoint)
    try:
        return _build_set(tracks, sr, blend_bars, phrase_bars, progress, use_stems,
                          _emit, accumulate, on_skip, play_bars)
    finally:
        _CTX_LOG.reset(tok_log)
        _CTX_CHECK.reset(tok_chk)


def _build_set(tracks, sr, blend_bars, phrase_bars, progress, use_stems, _emit, accumulate, on_skip,
               play_bars=None):
    n = len(tracks)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), []

    def skip(i: int, reason: str) -> None:
        _log().warning("skipping track %d: %s", i, reason)
        if on_skip:
            on_skip(i, reason)

    def load(i: int):
        _checkpoint()
        y = _resolve_track(tracks[i])
        if y.size == 0 or not np.isfinite(y).all():
            raise ValueError("empty or invalid audio")
        g = build_phrase_grid(y, sr, phrase_bars=phrase_bars)
        return y * np.float32(g.gain), g

    def track_marker(i: int, g: PhraseGrid, start: float, end: float) -> Dict:
        return {"type": "track", "index": i, "start_sec": start, "end_sec": end,
                "bpm": round(g.tempo, 1), "key": key_name(g), "camelot": g.camelot}

    # first loadable track
    ci, cur, grid_a = 0, None, None
    while ci < n:
        try:
            cur, grid_a = load(ci)
            break
        except Exception as e:
            skip(ci, f"could not load/analyse ({type(e).__name__}: {e})")
            ci += 1
    if cur is None:
        return np.zeros((0, 2), dtype=np.float32), []

    timeline: List[np.ndarray] = []
    markers: list = []
    resume = 0.0                      # seconds into `cur` where original playback resumes
    lead = np.zeros((0, 2), dtype=np.float32)   # audio that completes cur's mix-in
    pos = 0.0                         # running output position (seconds) — exact
    recent: List[str] = []

    def out(part, marker=None):
        if accumulate and len(part):
            timeline.append(part)
        _emit(part, marker)

    for ni in range(ci + 1, n):
        try:
            nxt, grid_b = load(ni)
        except Exception as e:
            skip(ni, f"could not load/analyse ({type(e).__name__}: {e})")
            continue
        try:
            window = None
            if play_bars:
                lo, hi = play_bars
                window = (lo, hi) if markers else (lo + 16, hi + 32)   # first track: longer
            plan = plan_transition(grid_a, grid_b, blend_bars=blend_bars, recent_types=recent,
                                   min_start_sec=resume, use_stems=use_stems, play_window=window)
            if window and plan.family == "fade" and "room" in plan.reason:
                # nothing fits the window (short/odd track): fall back to a normal mix-out
                plan = plan_transition(grid_a, grid_b, blend_bars=blend_bars, recent_types=recent,
                                       min_start_sec=resume, use_stems=use_stems)
        except Exception as e:
            skip(ni, f"no transition could be planned ({type(e).__name__}: {e})")
            continue

        # Body of the current track: original audio up to t1. Emitted BEFORE the
        # transition is rendered, so a client can play while the blend computes.
        r0 = int(round(resume * sr))
        a_t1 = max(r0, int(round(plan.a_t1_sec * sr)))
        body = cur[r0:a_t1]
        blen = (len(lead) + len(body)) / sr
        bmark = track_marker(ci, grid_a, pos, pos + blen)
        markers.append(bmark)
        first = True
        for part in (lead, body):
            if len(part):
                out(part, bmark if first else None)
                first = False
        if first:                                   # nothing audible, still report the marker
            out(np.zeros((0, 2), dtype=np.float32), bmark)
        pos += blen

        try:
            overlap, glide, resume_b = _render_transition(cur, grid_a, nxt, grid_b, plan, sr,
                                                          use_stems, recent=recent)
        except Exception as e:
            # the planned move cannot be rendered on this pair at all: a plain fade from
            # exactly where A's body stopped, so the timeline stays contiguous
            _log().warning("transition %d->%d failed (%s: %s); using a crossfade",
                           ci, ni, type(e).__name__, e)
            L = int(min(8.0 * sr, max(0, len(cur) - a_t1), len(nxt)))
            fo, fi = _equal_power(max(L, 1))
            overlap = _limit((_fit_len(cur[a_t1:a_t1 + L], L) * fo[:L, None]
                              + _fit_len(nxt[:L], L) * fi[:L, None]).astype(np.float32), sr)
            glide, resume_b = np.zeros((0, 2), dtype=np.float32), L / sr
            plan.transition_type, plan.family, plan.blend_bars = "fade", "fade", 0
            plan.reason = "fallback crossfade (the planned move failed on this material)"
        recent.append(plan.transition_type)
        tmark = {"type": "transition", "from": ci, "to": ni,
                 "start_sec": pos, "end_sec": pos + len(overlap) / sr, **_marker_info(plan)}
        markers.append(tmark)
        out(overlap, tmark)
        pos += len(overlap) / sr

        cur, grid_a, resume, lead, ci = nxt, grid_b, resume_b, glide, ni
        if progress:
            progress(ni, n)

    tail = cur[int(round(resume * sr)):]
    tlen = (len(lead) + len(tail)) / sr
    tlmark = track_marker(ci, grid_a, pos, pos + tlen)
    markers.append(tlmark)
    first = True
    for part in (lead, tail):
        if len(part):
            out(part, tlmark if first else None)
            first = False

    if accumulate:
        final = np.vstack(timeline) if timeline else np.zeros((0, 2), dtype=np.float32)
        return _headroom(final).astype(np.float32), markers
    return np.zeros((0, 2), dtype=np.float32), markers
