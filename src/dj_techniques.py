"""
DJ technique library — the moves the club mixer can play.

Everything here works on audio the engine has ALREADY put on a common beat grid
(src/club_mixer.py warps both decks onto one master tempo curve), so a technique
never has to think about timing drift: it only decides WHO plays WHAT, WHEN, in
bars. This is where the ideas from the original technique executor / stem
orchestrator live on (layered reveal, counter-melody, vocal overlay handoff,
"bass from A, beat from B", call-and-response, back-and-forth cuts, drum roll,
backspin, washout, echo out, drop on the one), rebuilt on bar-accurate timing.

Three families:

  beat-matched, STEMS (demucs: drums / bass / other / vocals per deck)
      stem_blend      one kit + one bassline at a time, melodies blend, vocals never overlap
      layered_reveal  B arrives stem by stem while A is stripped stem by stem
      beat_swap       B's drums slide in under A's song, then the rest follows
      vocal_ride      A's vocal keeps riding over B's instrumental (live mashup)
      vocal_tease     B's vocal is teased over A's instrumental before the switch
      drum_trade      the two drum kits trade bars (call and response), then B takes over
      stem_swap       keys clash: nothing melodic overlaps, everything swaps on the "1"

  beat-matched, EQ only (3 bands per deck — no stems needed)
      blend / filter / swap / drop_swap (kill the bass, build, drop B) / chop (transform cuts)

  cuts (tempos too far apart to beat-match — make it an event instead)
      roll_drop (loop roll 1 → 1/2 → 1/4 → 1/8 + filter, then drop), spinback,
      brake (tape stop), washout (reverb tail), echo_out
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as sig

HP = np.pi / 2.0

# The club engine's own bar-timed moves. The project's ORIGINAL library (23 techniques
# + 8 stem conversations, see src/technique_bridge.py) is the main rotation; these fill
# the gaps it does not cover.
STEM_TECHNIQUES = ("stem_blend", "beat_swap", "vocal_ride", "vocal_tease", "drum_trade", "stem_swap")
EQ_TECHNIQUES = ("blend", "filter", "swap", "drop_swap", "chop")
CUT_TECHNIQUES = ("roll_drop", "spinback", "brake", "echo_out")
# bars of the OUTGOING track each cut reshapes before the switch / rings on after it.
# Original cut-capable techniques work on a 2-bar segment that starts at the switch.
CUT_LEAD_BARS = {"roll_drop": 2.0, "spinback": 0.5, "brake": 0.5, "echo_out": 0.0}
CUT_TAIL_BARS = {"roll_drop": 1.0, "spinback": 1.0, "brake": 1.0, "echo_out": 2.0}
ORIGINAL_CUT_BARS = 2.0
# cuts that hit hardest when the new track comes in on its first big moment
CUT_WANTS_DROP = {"roll_drop", "spinback", "brake", "backspin", "drop_on_the_one", "quick_cut"}

LABELS = {
    "stem_blend": "stem blend", "beat_swap": "beat swap",
    "vocal_ride": "vocal ride", "vocal_tease": "vocal tease", "drum_trade": "drum trade",
    "stem_swap": "stem swap", "blend": "eq blend", "filter": "filter blend",
    "swap": "bass + mid swap", "drop_swap": "build + drop", "chop": "transform cuts",
    "roll_drop": "loop roll + drop", "spinback": "spinback", "brake": "tape stop",
    "echo_out": "echo out", "washout": "reverb washout", "fade": "crossfade",
}
DESCRIPTIONS = {
    "stem_blend": "one kit and one bassline at a time; melodies blend, vocals never overlap",
    "beat_swap": "next track's drums slide in under this song, then the rest follows",
    "vocal_ride": "this track's vocal keeps riding over the next track's instrumental",
    "vocal_tease": "next track's vocal teased over this track's instrumental",
    "drum_trade": "the two drum kits trade bars, then the next one takes over",
    "stem_swap": "keys clash — nothing melodic overlaps, everything swaps on the downbeat",
    "blend": "long EQ blend, bass swapped on the phrase downbeat",
    "filter": "EQ blend with the outgoing track filtered away",
    "swap": "keys clash — hats teased, then mids and bass swap on the downbeat",
    "drop_swap": "bass killed, filter build, next track dropped on the downbeat",
    "chop": "rhythmic cuts between the decks, then the next track takes over",
    "roll_drop": "loop roll (1 → 1/2 → 1/4 → 1/8) with a rising filter, then the drop",
    "spinback": "spinback, then the next track drops on its first beat",
    "brake": "tape stop, then the next track drops on its first beat",
    "washout": "reverb wash rings out over the next track's intro",
    "echo_out": "echoed out on the phrase end; next track drops on its first beat",
    "fade": "no steady beat to lock onto — smooth crossfade",
}


# ----------------------------------------------------------------------------- #
# Small DSP helpers
# ----------------------------------------------------------------------------- #
def as_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.column_stack([y, y])
    if y.shape[1] == 1:
        return np.column_stack([y[:, 0], y[:, 0]])
    return y[:, :2]


def fit_len(seg: np.ndarray, n: int) -> np.ndarray:
    """Truncate or zero-pad a stereo segment to exactly n samples."""
    if len(seg) >= n:
        return seg[:n]
    pad = np.zeros((n - len(seg), seg.shape[1]), dtype=seg.dtype)
    return np.vstack([seg, pad])


def _g(x: np.ndarray) -> np.ndarray:
    return x[:, None]


def ramp(n: int, start: float, end: float) -> np.ndarray:
    """0 before `start`, 1 after `end`, linear in between (sample positions)."""
    if end <= start:
        return (np.arange(n) >= end).astype(float)
    return np.clip((np.arange(n) - start) / (end - start), 0.0, 1.0)


def band_split(seg: np.ndarray, sr: int, lo: float = 250.0, hi: float = 2500.0):
    """3-band split with perfect reconstruction (low + mid + high == seg)."""
    nyq = sr / 2.0
    sos_lo = sig.butter(4, min(lo / nyq, 0.99), btype='low', output='sos')
    sos_hi = sig.butter(4, min(hi / nyq, 0.99), btype='high', output='sos')
    low = sig.sosfiltfilt(sos_lo, seg, axis=0)
    high = sig.sosfiltfilt(sos_hi, seg, axis=0)
    return low, seg - low - high, high


def _sweep(x: np.ndarray, sr: int, pos: np.ndarray, cutoffs, btype: str) -> np.ndarray:
    stages = [x]
    for fc in cutoffs:
        sos = sig.butter(2, min(fc / (sr / 2.0), 0.99), btype=btype, output='sos')
        stages.append(sig.sosfiltfilt(sos, x, axis=0))
    p = np.clip(pos, 0.0, 1.0) * (len(stages) - 1)
    out = np.zeros_like(x)
    for i, st in enumerate(stages):
        out += st * _g(np.clip(1.0 - np.abs(p - i), 0.0, 1.0))
    return out


def lp_sweep(x: np.ndarray, sr: int, pos: np.ndarray) -> np.ndarray:
    """A DJ filter knob turned down: pos 0 = open, 1 = closed to ~300 Hz."""
    return _sweep(x, sr, pos, (5000.0, 1500.0, 300.0), 'low')


def hp_sweep(x: np.ndarray, sr: int, pos: np.ndarray) -> np.ndarray:
    """High-pass build: pos 0 = full range, 1 = only the top (~3 kHz up)."""
    return _sweep(x, sr, pos, (250.0, 900.0, 3000.0), 'high')


def echo_tail(src: np.ndarray, n: int, delay: int, sr: int, feedback: float = 0.55,
              first: float = 0.6, hp: float = 300.0) -> np.ndarray:
    """Delay-throw tail: `src` (the last slice before a cut) repeats every `delay`
    samples, thinner and quieter each time. Returns n samples starting AT the cut."""
    tail = np.zeros((n, 2), dtype=np.float32)
    src = as_stereo(np.asarray(src, dtype=np.float32))
    if len(src) < 64 or n <= 0:
        return tail
    sos = sig.butter(2, min(hp / (sr / 2.0), 0.99), btype='high', output='sos')
    src = sig.sosfilt(sos, src, axis=0).astype(np.float32)
    f = max(1, min(len(src) // 2, int(0.008 * sr)))
    src[:f] *= np.linspace(0, 1, f)[:, None]
    src[-f:] *= np.linspace(1, 0, f)[:, None]
    gain, i0 = first, 0
    while i0 < n and gain > 0.012:
        m = min(len(src), n - i0)
        tail[i0:i0 + m] += gain * src[:m]
        gain *= feedback
        i0 += max(1, delay)
    tail *= _g(1.0 - ramp(n, 0.7 * n, n - 1)).astype(np.float32)   # always silent by the end
    return tail


_IR_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def reverb_ir(sr: int, rt60: float = 2.6) -> np.ndarray:
    """Dense stereo hall tail (decaying noise, high-passed, slightly darkening)."""
    key = (sr, rt60)
    if key not in _IR_CACHE:
        rng = np.random.default_rng(7)
        n = int(rt60 * sr)
        t = np.arange(n) / sr
        ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60)[:, None]
        ir = sig.sosfilt(sig.butter(2, 350.0 / (sr / 2.0), btype='high', output='sos'), ir, axis=0)
        ir = sig.sosfilt(sig.butter(1, 7000.0 / (sr / 2.0), btype='low', output='sos'), ir, axis=0)
        ir[:int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))[:, None]
        _IR_CACHE[key] = (ir / np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))).astype(np.float32)
    return _IR_CACHE[key]


def reverb_send(x: np.ndarray, sr: int, n_out: int, rt60: float = 2.6) -> np.ndarray:
    """100% wet reverb of stereo x, n_out samples long (so the tail can ring on)."""
    ir = reverb_ir(sr, rt60)
    wet = np.column_stack([sig.fftconvolve(x[:, c], ir[:, c]) for c in range(2)])
    return fit_len(wet.astype(np.float32), n_out)


def var_speed(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Vinyl-style playback along fractional sample positions (pitch follows speed)."""
    idx = np.arange(len(x))
    return np.column_stack([np.interp(pos, idx, x[:, c], left=0.0, right=0.0)
                            for c in range(x.shape[1])]).astype(np.float32)


# ----------------------------------------------------------------------------- #
# Bar-accurate timing of an overlap
# ----------------------------------------------------------------------------- #
@dataclass
class Timing:
    """Where every bar line of the overlap falls, in output samples. The tempo may
    be gliding from A's BPM to B's across the overlap, so bars are not equal length —
    techniques speak in bars and this converts."""
    N: int
    sr: int
    bar_pos: np.ndarray          # sample position of bar lines 0..pre+post
    pre: int                     # bars before the switch (B arriving under A)
    post: int                    # bars after the switch (A leaving under B)

    @property
    def S(self) -> int:          # the switch bar
        return self.pre

    @property
    def E(self) -> int:          # last bar line (end of the overlap)
        return self.pre + self.post

    def at(self, bar: float) -> float:
        nb = len(self.bar_pos) - 1
        b = float(np.clip(bar, 0.0, nb))
        return float(np.interp(b, np.arange(nb + 1), self.bar_pos))

    def beat(self, bar: Optional[float] = None) -> int:
        b = self.S if bar is None else bar
        lo, hi = max(0.0, b - 0.5), min(float(self.E), b + 0.5)
        return max(1, int(round((self.at(hi) - self.at(lo)) / (4.0 * max(hi - lo, 1e-6)))))

    def R(self, b0: float, b1: float) -> np.ndarray:
        return ramp(self.N, self.at(b0), self.at(b1))

    def fin(self, b0: float, b1: float) -> np.ndarray:      # equal-power fade in
        return np.sin(HP * self.R(b0, b1))

    def fout(self, b0: float, b1: float) -> np.ndarray:     # equal-power fade out
        return np.cos(HP * self.R(b0, b1))

    def cut_in(self, bar: float, beats: float = 1.0) -> np.ndarray:
        """In over the last `beats` before a bar line — owns the "1"."""
        return np.sin(HP * self.R(bar - beats / 4.0, bar))

    def cut_out(self, bar: float, beats: float = 1.0) -> np.ndarray:
        return np.cos(HP * self.R(bar - beats / 4.0, bar))

    def gate(self, b0: float, b1: float, edge_ms: float = 5.0) -> np.ndarray:
        """1 inside [b0, b1), 0 outside, with click-free edges."""
        e = edge_ms * self.sr / 1000.0
        a, b = self.at(b0), self.at(b1)
        return ramp(self.N, a - e, a) * (1.0 - ramp(self.N, b - e, b))

    def blend_w(self) -> np.ndarray:
        """0→1 dominance curve: slow at the edges, decisive around the switch."""
        n = np.arange(self.N)
        s2 = float(np.clip(self.at(self.S), 1, self.N - 1))
        u = np.clip(n / s2, 0.0, 1.0)
        v = np.clip((n - s2) / max(1.0, self.N - s2), 0.0, 1.0)
        return np.where(n < s2, 0.5 * u ** 1.5, 1.0 - 0.5 * (1.0 - v) ** 1.5)


def exit_with_echo(part: np.ndarray, x: float, t: Timing, beats: float = 0.5,
                   feedback: float = 0.5) -> np.ndarray:
    """Cut a part at sample x and throw its last half-beat into a delay."""
    x = int(np.clip(x, 0, t.N - 1))
    d = max(64, int(t.beat() * beats))
    out = part * _g(1.0 - ramp(t.N, x - 0.006 * t.sr, x))
    tail = echo_tail(part[max(0, x - d):x], t.N - x, d, t.sr, feedback=feedback, first=0.55)
    out[x:] += tail
    return out


def vocal_gap(voc: np.ndarray, target: float, lo: float, hi: float, sr: int) -> float:
    """Sample near `target` (within [lo, hi]) where the vocal takes a breath, so a
    vocal is never cut mid-word."""
    lo, hi = int(max(0, lo)), int(min(len(voc), hi))
    if hi - lo < int(0.2 * sr):
        return float(np.clip(target, lo, max(lo, hi)))
    w = int(0.04 * sr)
    env = np.sqrt(np.convolve(np.mean(voc[lo:hi] ** 2, axis=1), np.ones(w) / w, mode="same"))
    ref = float(np.percentile(env, 90)) + 1e-9
    cost = env / ref + 0.35 * np.abs(np.arange(lo, hi) - target) / max(1.0, hi - lo)
    return float(lo + int(np.argmin(cost)))


# ----------------------------------------------------------------------------- #
# Beat-matched techniques — EQ (3 bands)
# ----------------------------------------------------------------------------- #
def _bands(seg: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    low, mid, high = band_split(seg, sr)
    return {"low": low, "mid": mid, "high": high, "full": seg}


def _eq_blend(t: Timing, A, B, ctx, filt: bool = False) -> np.ndarray:
    wm = t.blend_w()
    wh = np.clip(1.25 * wm, 0.0, 1.0)                     # incoming highs lead
    a_top = A["mid"] * _g(np.cos(HP * wm)) + A["high"] * _g(np.cos(HP * wh))
    if filt:
        a_top = lp_sweep(a_top, t.sr, t.R(t.S / 2.0, t.S + 0.9 * t.post))
    return (A["low"] * _g(t.cut_out(t.S)) + B["low"] * _g(t.cut_in(t.S)) + a_top
            + B["mid"] * _g(np.sin(HP * wm)) + B["high"] * _g(np.sin(HP * wh)))


def _eq_swap(t: Timing, A, B, ctx) -> np.ndarray:
    v = t.R(t.S, t.E)
    b_high = np.maximum(0.85 * np.sin(HP * t.R(0, t.S)), t.cut_in(t.S, 2))
    return (A["low"] * _g(t.cut_out(t.S)) + B["low"] * _g(t.cut_in(t.S))
            + A["mid"] * _g(t.cut_out(t.S, 2)) + B["mid"] * _g(t.cut_in(t.S, 2))
            + A["high"] * _g(np.cos(HP * v ** 0.7)) + B["high"] * _g(b_high))


def _eq_drop_swap(t: Timing, A, B, ctx) -> np.ndarray:
    """Kill A's bass, sweep the filter up, then B lands on the downbeat at full weight."""
    s2 = t.at(t.S)
    build = max(1.0, min(2.0, float(t.pre)))
    a_top = hp_sweep(A["mid"] + A["high"], t.sr, 0.8 * t.R(t.S - build, t.S))
    a_top = exit_with_echo(a_top, s2, t, beats=0.5, feedback=0.45)
    drop = ramp(t.N, s2 - 0.004 * t.sr, s2)
    return A["low"] * _g(t.fout(t.S - build, t.S - build / 2.0)) + a_top + B["full"] * _g(drop)


def _chop(t: Timing, A, B, ctx) -> np.ndarray:
    """Transform cuts: the crossfader is thrown between the decks on the beat for the
    last two bars (A A A B | A B A B), then B takes over on the "1"."""
    bars = 2 if t.pre >= 2 else 1
    b0 = t.S - bars
    pattern = [0, 0, 0, 1, 0, 1, 0, 1] if bars == 2 else [0, 1, 0, 1]
    gb = np.zeros(t.N)
    for i, on in enumerate(pattern):
        if on:
            gb += t.gate(b0 + i / 4.0, b0 + (i + 1) / 4.0)
    gb = np.clip(gb + ramp(t.N, t.at(t.S) - 0.005 * t.sr, t.at(t.S)), 0.0, 1.0)
    return A["full"] * _g(1.0 - gb) + B["full"] * _g(gb)


# ----------------------------------------------------------------------------- #
# Beat-matched techniques — stems
# ----------------------------------------------------------------------------- #
def _stem_parts(seg: np.ndarray, stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    n = len(seg)
    p = {k: fit_len(as_stereo(np.asarray(stems.get(k, np.zeros((n, 2))), dtype=np.float32)), n)
         for k in ("drums", "bass", "other", "vocals")}
    # whatever the separator lost goes with the melodic stem, so the parts always
    # sum back to the original audio exactly (seamless joins with untouched audio)
    p["other"] = p["other"] + (seg - (p["drums"] + p["bass"] + p["other"] + p["vocals"]))
    sos = sig.butter(4, 4000.0 / (sr / 2.0), btype='high', output='sos')
    p["hats"] = sig.sosfiltfilt(sos, p["drums"], axis=0).astype(np.float32)   # cymbals / hats
    p["kit"] = p["drums"] - p["hats"]                                          # kick / snare body
    p["full"] = seg
    return p


def _hats(t: Timing, A, B, tease_to: Optional[float] = None, level: float = 0.8) -> np.ndarray:
    """B's hats sneak in before the switch; A's ring on a little after it."""
    end = t.S - max(1.0, t.pre / 4.0) if tease_to is None else tease_to
    b = np.maximum(level * t.fin(0, max(0.5, end)), t.cut_in(t.S))
    return B["hats"] * _g(b) + A["hats"] * _g(t.fout(t.S, t.S + max(1.0, t.post / 2.0)))


def _low_end_swap(t: Timing, A, B) -> np.ndarray:
    """One kit and one bassline at a time — swapped on the downbeat."""
    return ((A["kit"] + A["bass"]) * _g(t.cut_out(t.S)) + (B["kit"] + B["bass"]) * _g(t.cut_in(t.S)))


def _melody(t: Timing, A, B, ctx, a_hold: float = 0.0) -> np.ndarray:
    """The 'other' stems: a long blend when the keys agree (or a shorter one around
    the switch when a_hold is given), a quick swap on the "1" when they clash."""
    if not ctx.get("key_ok", True):
        return A["other"] * _g(t.cut_out(t.S, 2)) + B["other"] * _g(t.cut_in(t.S, 2))
    if a_hold > 0:
        w = t.R(t.S - 1.0, t.S + a_hold)
        return A["other"] * _g(np.cos(HP * w)) + B["other"] * _g(np.sin(HP * w))
    w = t.blend_w()
    a = lp_sweep(A["other"] * _g(np.cos(HP * w)), t.sr, 0.7 * t.R(t.S, t.S + 0.9 * t.post))
    return a + B["other"] * _g(np.sin(HP * w))


def _vocal_handoff(t: Timing, A, B, a_out_bar: Optional[float] = None,
                   b_in_bar: Optional[float] = None) -> np.ndarray:
    """A's vocal leaves on a breath (with an echo throw) before B's vocal enters —
    never two lead vocals at once."""
    a_bar = t.S if a_out_bar is None else a_out_bar
    x = vocal_gap(A["vocals"], t.at(a_bar), t.at(a_bar - 1.0), t.at(min(t.E - 0.25, a_bar + 0.5)), t.sr)
    b_bar = t.S if b_in_bar is None else b_in_bar
    start = max(t.at(b_bar), x)
    b_gain = np.sin(HP * ramp(t.N, start, min(t.N - 1, start + (t.at(t.E) - t.at(t.E - 1)))))
    return exit_with_echo(A["vocals"], x, t) + B["vocals"] * _g(b_gain)


def _stem_blend(t: Timing, A, B, ctx) -> np.ndarray:
    return _low_end_swap(t, A, B) + _hats(t, A, B) + _melody(t, A, B, ctx) + _vocal_handoff(t, A, B)


def _stem_swap(t: Timing, A, B, ctx) -> np.ndarray:
    return (_low_end_swap(t, A, B) + _hats(t, A, B) + _melody(t, A, B, {"key_ok": False})
            + _vocal_handoff(t, A, B))


def _beat_swap(t: Timing, A, B, ctx) -> np.ndarray:
    """B's drums replace A's half-way through the lead-in — A's song over B's groove —
    then bass, melody and vocal follow on the switch."""
    d = max(1.0, float(t.pre // 2))
    drums = (A["drums"] * _g(t.cut_out(d)) + B["kit"] * _g(t.cut_in(d))
             + B["hats"] * _g(np.maximum(0.7 * t.fin(0, d), t.cut_in(d))))
    bass = A["bass"] * _g(t.cut_out(t.S)) + B["bass"] * _g(t.cut_in(t.S))
    return drums + bass + _melody(t, A, B, ctx, a_hold=min(2.0, t.post)) + _vocal_handoff(t, A, B)


def _vocal_ride(t: Timing, A, B, ctx) -> np.ndarray:
    """A's vocal keeps going after the switch, over B's drums, bass and music —
    a live mashup — then hands over to B's own vocal on a breath."""
    w = t.R(t.S - 0.5, t.S + 1.0)
    music = A["other"] * _g(np.cos(HP * w)) + B["other"] * _g(np.sin(HP * w))
    ride_to = t.E - min(1.0, t.post / 4.0)
    x = vocal_gap(A["vocals"], t.at(ride_to), t.at(t.S + t.post / 2.0), t.at(t.E - 0.25), t.sr)
    vocals = exit_with_echo(A["vocals"], x, t) + B["vocals"] * _g(np.sin(HP * ramp(t.N, x, t.N - 1)))
    return _low_end_swap(t, A, B) + _hats(t, A, B) + music + vocals


def _vocal_tease(t: Timing, A, B, ctx) -> np.ndarray:
    """A drops to its instrumental and B's vocal is teased over it; the rest of B
    arrives on the switch."""
    w = t.R(t.S - 0.5, t.S + 1.5)
    music = A["other"] * _g(np.cos(HP * w)) + B["other"] * _g(np.sin(HP * w))
    x = vocal_gap(A["vocals"], t.at(1.0), t.at(0.25), t.at(min(2.0, t.S - 0.5)), t.sr)
    vocals = exit_with_echo(A["vocals"], x, t, feedback=0.4) \
        + B["vocals"] * _g(np.sin(HP * ramp(t.N, x, x + t.beat() * 2)))
    return _low_end_swap(t, A, B) + _hats(t, A, B) + music + vocals


def _drum_trade(t: Timing, A, B, ctx) -> np.ndarray:
    """Call and response: the two kits trade bars before the switch, then B keeps it."""
    step = 1.0 if t.pre >= 4 else 0.5
    b0 = t.S - 4 * step
    gb = t.gate(b0 + step, b0 + 2 * step) + t.gate(b0 + 3 * step, t.S + 0.01)
    gb = np.clip(gb + ramp(t.N, t.at(t.S) - 0.005 * t.sr, t.at(t.S)), 0.0, 1.0)
    drums = A["drums"] * _g(1.0 - gb) + B["drums"] * _g(gb)
    bass = A["bass"] * _g(t.cut_out(t.S)) + B["bass"] * _g(t.cut_in(t.S))
    return drums + bass + _melody(t, A, B, ctx) + _vocal_handoff(t, A, B)


_STEM_FUNCS = {"stem_blend": _stem_blend, "beat_swap": _beat_swap, "vocal_ride": _vocal_ride, "vocal_tease": _vocal_tease,
               "drum_trade": _drum_trade, "stem_swap": _stem_swap, "chop": _chop}
_EQ_FUNCS = {"blend": _eq_blend, "filter": lambda t, A, B, c: _eq_blend(t, A, B, c, filt=True),
             "swap": _eq_swap, "drop_swap": _eq_drop_swap, "chop": _chop}


def stem_features(SA: Dict[str, np.ndarray], SB: Dict[str, np.ndarray], t: Timing) -> Dict:
    """What is actually playing in each deck, before and after the switch."""
    def active(P, key, b0, b1, thr):
        bars = np.arange(int(b0), int(b1))
        if len(bars) == 0:
            return 0.0
        hits = 0
        for b in bars:
            i0, i1 = int(t.at(b)), int(t.at(b + 1))
            mix = float(np.sqrt(np.mean(P["full"][i0:i1] ** 2)) + 1e-9)
            hits += float(np.sqrt(np.mean(P[key][i0:i1] ** 2))) > thr * mix
        return hits / len(bars)
    def level(P, b0, b1):
        i0, i1 = int(t.at(b0)), max(int(t.at(b0)) + 1, int(t.at(b1)))
        return 20.0 * np.log10(float(np.sqrt(np.mean(P["full"][i0:i1] ** 2))) + 1e-9)
    return {
        "va_pre": active(SA, "vocals", 0, t.S, 0.25), "va_post": active(SA, "vocals", t.S, t.E, 0.25),
        "vb_pre": active(SB, "vocals", 0, t.S, 0.25), "vb_post": active(SB, "vocals", t.S, t.E, 0.25),
        "da": active(SA, "drums", 0, t.E, 0.2), "db": active(SB, "drums", 0, t.E, 0.2),
        "db_pre": active(SB, "drums", 0, t.S, 0.2),
        # how loud B is next to A, before and after the switch (dB)
        "lb_pre": level(SB, 0, t.S) - level(SA, 0, t.S),
        "lb_post": level(SB, t.S, t.E) - level(SA, 0, t.S),
    }


# Techniques that hand the groove / the bed to B BEFORE the switch. They need B to
# really be playing there — over a quiet, drumless intro they leave dead air.
NEEDS_B_EARLY = {"staggered_stem_mix", "partial_stem_separation", "vocal_layering", "acapella_overlay",
                 "conv_bass_from_a_beat_from_b", "conv_melody_a_drums_vocals_b", "conv_call_response",
                 "conv_counter_melody", "conv_interweave", "beat_swap", "drum_trade",
                 "back_and_forth", "chop", "double_drop"}


def hole_db(out: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray, sr: int) -> float:
    """Worst 2-second level drop of a rendered transition relative to a plain
    equal-power crossfade of the same two decks (dB, <= 0). A technique may thin the
    sound out on purpose, but it must not leave seconds of dead air."""
    n = min(len(out), len(seg_a), len(seg_b))
    w = np.linspace(0.0, 1.0, n)[:, None]
    ref = seg_a[:n] * np.cos(HP * w) + seg_b[:n] * np.sin(HP * w)
    win, hop = int(2.0 * sr), int(0.5 * sr)
    if n < win + hop:
        return 0.0
    worst = 0.0
    for i in range(0, n - win, hop):
        a = float(np.sqrt(np.mean(out[i:i + win] ** 2))) + 1e-9
        b = float(np.sqrt(np.mean(ref[i:i + win] ** 2))) + 1e-9
        if b > 2e-3:                                   # ignore stretches where both decks are silent
            worst = min(worst, 20.0 * np.log10(a / b))
    return worst


def _pick(cands: List[Tuple[str, float]], recent: List[str]) -> str:
    """Highest score wins, but a DJ doesn't play the same move twice running."""
    best, best_s = cands[0][0], -1e9
    for name, s in cands:
        if recent and recent[-1] == name:
            s -= 1.0
        s -= 0.45 * sum(1 for r in recent[-4:-1] if r == name)
        s -= 0.25 * sum(1 for r in recent if r == name)      # whole-set variety
        if s > best_s + 1e-9:
            best, best_s = name, s
    return best


def choose_eq_technique(key_ok: bool, pre: int, b_has_drop: bool, recent: List[str],
                        exclude=()) -> str:
    """No stems: the original full-mix techniques first, the club EQ moves as extras."""
    c: List[Tuple[str, float]] = []
    if key_ok:
        c += [("bass_swap", 0.66), ("long_blend", 0.6), ("filter_sweep", 0.6), ("blend", 0.58),
              ("filter", 0.52), ("energy_build", 0.5), ("breakdown_to_build", 0.45), ("drop_mix", 0.45)]
    else:
        c += [("swap", 0.62), ("drop_on_the_one", 0.55), ("quick_cut", 0.45)]
    if pre >= 2:
        c += [("back_and_forth", 0.55), ("loop_transition", 0.5), ("drum_roll", 0.5), ("chop", 0.45)]
    if b_has_drop and pre >= 2:
        c += [("double_drop", 0.6 if key_ok else 0.5), ("drop_swap", 0.5)]
    c = [(n, s) for n, s in c if n not in exclude]
    return _pick(c or [("blend" if key_ok else "swap", 0.5)], recent)


def choose_stem_technique(f: Dict, key_ok: bool, pre: int, post: int, recent: List[str],
                          b_has_drop: bool = False, exclude=()) -> str:
    """With stems: pick from what is really playing. The original stem conversations
    and stem techniques lead; melodic overlaps are only offered when the keys agree."""
    drums = f["da"] >= 0.5 and f["db"] >= 0.5
    a_sings_after, a_sings = f["va_post"] >= 0.5, max(f["va_pre"], f["va_post"]) >= 0.5
    b_sings_before, b_quiet_after = f["vb_pre"] >= 0.5, f["vb_post"] <= 0.34
    c: List[Tuple[str, float]] = [("staggered_stem_mix", 0.74), ("partial_stem_separation", 0.7),
                                  ("conv_interweave", 0.7 if key_ok else 0.55)]
    if key_ok:
        c += [("conv_layered_reveal", 0.8), ("conv_progressive_morph", 0.6), ("stem_blend", 0.6),
              ("progressive_morph", 0.5)]
        if a_sings_after:
            c += [("conv_counter_melody", 0.84), ("acapella_overlay", 0.78)]
            if b_quiet_after and post >= 4:
                c.append(("vocal_ride", 0.8))
        if b_sings_before:
            c.append(("conv_vocal_overlay_handoff", 0.84))
            if pre >= 4:
                c.append(("vocal_tease", 0.74))
        if drums:
            c.append(("conv_melody_a_drums_vocals_b", 0.78))
    else:
        c.append(("stem_swap", 0.72))
    if a_sings:
        c.append(("vocal_layering", 0.74))
    if drums:
        c += [("conv_bass_from_a_beat_from_b", 0.78), ("conv_call_response", 0.72)]
        if pre >= 4:
            c += [("beat_swap", 0.76), ("drum_trade", 0.7)]
    if pre >= 2:
        c += [("back_and_forth", 0.5), ("chop", 0.42)]
    if b_has_drop and pre >= 2:
        c.append(("double_drop", 0.55))
    b_present_early = f.get("db_pre", 1.0) >= 0.5 and f.get("lb_pre", 0.0) >= -10.0
    c = [(n, s) for n, s in c if n not in exclude and (b_present_early or n not in NEEDS_B_EARLY)]
    return _pick(c or [("stem_blend" if key_ok else "stem_swap", 0.5)], recent)


def choose_cut_technique(a_energy_high: bool, room_bars: float, recent: List[str],
                         b_cue_hot: bool = True) -> str:
    """Tempos cannot be matched: pick an exit that makes the change an event. A hard
    drop only works if the next track comes in with weight — onto a quiet intro,
    let the old track ring out instead."""
    if not b_cue_hot:
        return _pick([("washout", 0.8), ("echo_out", 0.78)], recent)
    c = ([("backspin", 0.8), ("roll_drop", 0.78), ("drop_on_the_one", 0.7), ("echo_out", 0.68),
          ("drum_roll", 0.66), ("spinback", 0.62), ("washout", 0.55), ("brake", 0.5), ("quick_cut", 0.45)]
         if a_energy_high else
         [("washout", 0.8), ("echo_out", 0.78), ("backspin", 0.65), ("roll_drop", 0.6),
          ("drop_on_the_one", 0.55), ("brake", 0.5), ("drum_roll", 0.5), ("spinback", 0.45)])
    c = [(n, s) for n, s in c if CUT_LEAD_BARS.get(n, 0.0) <= room_bars] or [("echo_out", 0.5)]
    return _pick(c, recent)


def render_beatmatched(technique: str, seg_a: np.ndarray, seg_b: np.ndarray, t: Timing,
                       ctx: Dict, stems_a: Optional[Dict] = None, stems_b: Optional[Dict] = None,
                       parts: Optional[Tuple[Dict, Dict]] = None) -> np.ndarray:
    """Render one beat-matched overlap. seg_a / seg_b are the two decks, already on
    the same beat grid and exactly t.N samples long."""
    if parts is not None:
        A, B = parts
        fn = _STEM_FUNCS[technique]
    elif stems_a is not None and stems_b is not None and technique in _STEM_FUNCS:
        A, B = _stem_parts(seg_a, stems_a, t.sr), _stem_parts(seg_b, stems_b, t.sr)
        fn = _STEM_FUNCS[technique]
    else:
        A, B = _bands(seg_a, t.sr), _bands(seg_b, t.sr)
        fn = _EQ_FUNCS.get(technique, _eq_blend)
    out = fn(t, A, B, ctx).astype(np.float32)
    # Pin the edges: the overlap must begin as pure A and end as pure B so it joins
    # the untouched audio on both sides without a seam (a no-op for a correct score).
    k = min(t.N // 4, int(0.02 * t.sr))
    if k > 1:
        r = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        out[:k] = seg_a[:k] * (1 - r) + out[:k] * r
        out[-k:] = out[-k:] * (1 - r) + seg_b[-k:] * r
    return out


# ----------------------------------------------------------------------------- #
# Cuts — for pairs that cannot be beat-matched
# ----------------------------------------------------------------------------- #
def _loop_roll(a: np.ndarray, beat: int, sr: int) -> np.ndarray:
    """8 beats: the first beat looped x4, then 1/2-beat x4, 1/4-beat x4, 1/8-beat x8,
    with the filter opening upwards — the classic build before a drop."""
    L = len(a)
    out = np.zeros_like(a)
    src = a[:beat]
    plan = [(1.0, 4), (0.5, 4), (0.25, 4), (0.125, 8)]
    pos = 0
    for frac, reps in plan:
        n = max(32, int(round(beat * frac)))
        sl = src[:n].copy()
        f = max(1, min(n // 4, int(0.003 * sr)))
        sl[:f] *= np.linspace(0, 1, f)[:, None]
        sl[-f:] *= np.linspace(1, 0, f)[:, None]
        for _ in range(reps):
            if pos >= L:
                break
            m = min(n, L - pos)
            out[pos:pos + m] = sl[:m]
            pos += n
    out[:beat] = a[:beat]                                   # first pass is the untouched audio
    out = hp_sweep(out, sr, ramp(L, 0.15 * L, L) ** 1.5)
    return out * _g(1.0 + 0.25 * ramp(L, 0.5 * L, L))


def render_cut(technique: str, cur: np.ndarray, a1: int, sw: int, nxt: np.ndarray, b1: int,
               tail_n: int, beat: int, sr: int) -> np.ndarray:
    """Render a cut: A[a1:sw] reshaped by the effect, then B drops in at `sw` on its
    cue `b1` while the effect's tail rings out. Returns (sw - a1) + tail_n samples."""
    L = max(0, sw - a1)
    N = L + tail_n
    out = np.zeros((N, 2), dtype=np.float32)
    seg_b = fit_len(nxt[b1:b1 + tail_n], tail_n).copy()
    k = min(tail_n, int(0.003 * sr))
    if k > 1:
        seg_b[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    out[L:] += seg_b
    a = fit_len(cur[a1:sw], L).astype(np.float32) if L else np.zeros((0, 2), dtype=np.float32)

    if technique == "roll_drop" and L > 8 * 64:
        roll = _loop_roll(a, max(64, L // 8), sr)
        out[:L] += roll
        out[L:] += echo_tail(roll[-max(64, L // 32):], tail_n, max(64, L // 32), sr,
                             feedback=0.6, first=0.35, hp=1500.0)
    elif technique == "spinback" and L > 256:
        # the record is pulled backwards, faster and faster
        tt = np.arange(L) / L
        back = np.cumsum(1.0 + 9.0 * tt ** 1.6)
        fx = var_speed(cur[max(0, a1 - int(back[-1]) - 2):a1 + 1],
                       (a1 - max(0, a1 - int(back[-1]) - 2)) - back)
        out[:L] += fx * _g(1.0 - 0.9 * ramp(L, 0.7 * L, L))
    elif technique == "brake" and L > 256:
        # the platter is stopped: speed and pitch fall to zero
        tt = np.arange(L) / L
        fwd = np.cumsum((1.0 - tt) ** 1.3)
        fx = var_speed(cur[a1:a1 + L + 2], fwd)
        out[:L] += fx * _g(1.0 - ramp(L, 0.8 * L, L))
    else:                                                   # echo_out (also the safe default)
        if L:
            out[:L] += a
        out[L:] += echo_tail(cur[max(0, sw - beat):sw], tail_n, max(1, beat // 2), sr)
    return out
