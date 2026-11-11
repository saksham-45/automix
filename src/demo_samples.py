"""
Royalty-free demo tracks, synthesized on demand.

The public demo must not ship copyrighted music (that would defeat both the
security/legal posture and the research framing). Instead we generate three short
electronic tracks with REAL SONG STRUCTURE — drums-only intro, build, drop,
breakdown, second drop, drums-only outro — at distinct tempos and keys, so the
mixer has what a DJ actually works with: an intro to mix in over, an outro to mix
out of, phrase boundaries to land on, and keys that do / don't get along.

`ensure_demo_samples()` is idempotent: it writes the files only if missing (or if
an older, structure-less version is found), so it is cheap to call at container
build time and again at server startup as a fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# (filename, bpm, root MIDI note of the minor key)
#   a -> b is a compatible key move (A minor -> E minor, one step on the Camelot
#   wheel) at +2 BPM: a long melodic blend. b -> c clashes (E minor -> F# minor is
#   two steps away), so the mixer should keep the melodies apart and swap instead.
_SPECS = [
    ("sample_a.wav", 122.0, 45),   # A minor
    ("sample_b.wav", 124.0, 40),   # E minor
    ("sample_c.wav", 128.0, 42),   # F# minor
]

# (name, bars) — 64 bars total, every boundary on an 8-bar phrase.
_SECTIONS = [("intro", 8), ("build", 8), ("drop", 16), ("break", 8), ("drop", 16), ("outro", 8)]
_MIN_SECONDS = 90.0   # anything shorter on disk is the old loop-only demo -> regenerate

# i - VI - III - VII in natural minor, as semitone offsets from the root.
_PROG = [(0, 3, 7), (8, 12, 15), (3, 7, 10), (10, 14, 17)]


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _add(out: np.ndarray, i0: int, x: np.ndarray) -> None:
    if i0 >= len(out) or i0 < 0:
        return
    n = min(len(x), len(out) - i0)
    out[i0:i0 + n] += x[:n]


def _synth_track(bpm: float, root_midi: int, sr: int) -> np.ndarray:
    rng = np.random.default_rng(int(bpm * 10) + root_midi)   # deterministic per track
    beat = 60.0 / bpm
    bar = 4 * beat
    sixteenth = beat / 4
    total_bars = sum(b for _, b in _SECTIONS)
    n = int(total_bars * bar * sr) + sr // 2
    out = np.zeros(n, dtype=np.float64)

    # --- one-shots ---------------------------------------------------------- #
    kn = int(0.22 * sr)
    kick = np.exp(-np.linspace(0, 7, kn)) * np.sin(
        2 * np.pi * np.cumsum(np.linspace(150, 42, kn) ** 1.0) / sr)
    kick[:int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))

    def noise_burst(dur, decay, hp):
        m = int(dur * sr)
        x = rng.standard_normal(m) * np.exp(-np.linspace(0, decay, m))
        # cheap high-pass: subtract a moving average
        k = max(2, int(sr / hp))
        return x - np.convolve(x, np.ones(k) / k, mode="same")

    def tone(freq, dur, decay, harmonics=(1.0,), attack=0.004):
        m = int(dur * sr)
        t = np.arange(m) / sr
        x = sum(a * np.sin(2 * np.pi * freq * (h + 1) * t) for h, a in enumerate(harmonics))
        env = np.exp(-np.linspace(0, decay, m))
        a = max(1, int(attack * sr))
        env[:a] *= np.linspace(0, 1, a)
        rel = max(1, int(0.01 * sr))
        env[-rel:] *= np.linspace(1, 0, rel)
        return x * env

    saw = (1.0, 0.5, 0.33, 0.25, 0.2, 0.16)

    bar_idx = 0
    for name, bars in _SECTIONS:
        for b in range(bars):
            t_bar = (bar_idx + b) * bar
            chord = _PROG[(bar_idx + b) % len(_PROG)]
            last_bar = (b == bars - 1)
            drums = name in ("intro", "build", "drop", "outro")
            full = name == "drop"

            for q in range(4):                       # beats
                tb = t_bar + q * beat
                if drums:
                    _add(out, int(tb * sr), 0.95 * kick)
                    # offbeat open-ish hat
                    _add(out, int((tb + beat / 2) * sr), 0.16 * noise_burst(0.07, 7, 6000))
                if name in ("build", "drop", "outro") and q in (1, 3):
                    _add(out, int(tb * sr), 0.22 * noise_burst(0.16, 6, 1500))   # clap
            if full:
                for s in range(16):                  # 16th closed hats
                    _add(out, int((t_bar + s * sixteenth) * sr),
                         0.05 * noise_burst(0.03, 9, 8000))

            # bass: off-beat 8ths on the chord root (classic house), build + drops
            if name in ("build", "drop"):
                f = _hz(root_midi + chord[0] - (12 if chord[0] >= 8 else 0))
                for q in range(4):
                    _add(out, int((t_bar + (q + 0.5) * beat) * sr),
                         0.42 * tone(f, beat * 0.48, 2.2, (1.0, 0.35, 0.12)))

            # chords: syncopated stabs in the drops, a sustained pad in the breakdown
            if full:
                for s in (0, 3, 6, 10, 12):
                    x = sum(tone(_hz(root_midi + 24 + iv), 0.28, 5, saw) for iv in chord)
                    _add(out, int((t_bar + s * sixteenth) * sr), 0.045 * x)
            if name == "break":
                x = sum(tone(_hz(root_midi + 24 + iv), bar, 0.6, (1.0, 0.4, 0.2), attack=0.25)
                        for iv in chord)
                _add(out, int(t_bar * sr), 0.07 * x)

            # lead arpeggio: drops + breakdown (+ quietly in the build's last 4 bars)
            if full or name == "break" or (name == "build" and b >= bars - 4):
                lvl = 0.06 if full else 0.045
                for s in range(0, 16, 2):
                    iv = chord[(s // 2) % 3] + (12 if (s // 2) % 4 == 3 else 0)
                    _add(out, int((t_bar + s * sixteenth) * sr),
                         lvl * tone(_hz(root_midi + 36 + iv), 0.2, 5, (1.0, 0.3)))

            # riser into the next section on the last bar of build / break
            if last_bar and name in ("build", "break"):
                m = int(bar * sr)
                r = rng.standard_normal(m) * np.linspace(0, 1, m) ** 2
                k = 24
                r = r - np.convolve(r, np.ones(k) / k, mode="same")
                _add(out, int(t_bar * sr), 0.10 * r)
        bar_idx += bars

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.89).astype(np.float32)


def ensure_demo_samples(out_dir, sr: int = 44100) -> list:
    """Write the demo tracks into out_dir if absent/outdated. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, bpm, root in _SPECS:
        p = out_dir / name
        stale = True
        if p.exists():
            try:
                stale = sf.info(str(p)).duration < _MIN_SECONDS
            except Exception:
                stale = True
        if stale:
            sf.write(str(p), _synth_track(bpm, root, sr), sr, subtype="PCM_16")
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/demo"
    for p in ensure_demo_samples(d):
        print(p)
