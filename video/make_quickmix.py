"""Render an automix "quick mix": each song plays for 4-8 bars after it's mixed in,
then a full 16-bar transition into the next one. Nothing inside a song is cut.

usage: python video/make_quickmix.py song1.wav song2.wav ... [--out qm]
writes <out>.wav (24-bit) and <out>.json (markers + track titles for the video HUD).
Titles come from the file names ("Artist - Title.wav"); edit the JSON if you want.
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import club_mixer as cm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("songs", nargs="+")
ap.add_argument("--out", default="qm")
ap.add_argument("--no-stems", action="store_true")
args = ap.parse_args()

SR = 44100


def meta(p):
    stem = Path(p).stem
    artist, _, title = stem.partition(" - ")
    return dict(title=title or stem, artist=artist if title else "")


tracks = [meta(p) for p in args.songs]
t0 = time.time()
audio, markers = cm.build_continuous_set(
    [lambda p=p: sf.read(p, dtype="float32")[0] for p in args.songs],
    sr=SR, blend_bars=16, use_stems=not args.no_stems, play_bars=(4, 8))
print(f"mix {len(audio) / SR / 60:.1f} min in {time.time() - t0:.0f}s, peak {np.abs(audio).max():.3f}")
for m in markers:
    ts = f"{int(m['start_sec'] // 60):2d}:{int(m['start_sec'] % 60):02d}"
    if m["type"] == "track":
        r = tracks[m["index"]]
        print(f"  {ts}  {r['artist']} — {r['title']}  ({m['bpm']} BPM {m['key']})")
    else:
        print(f"  {ts}    ↳ {m['label']} [{m['library']}{', stems' if m['stems'] else ''}] "
              f"{m['bars']} bars, {m['end_sec'] - m['start_sec']:.0f}s | {m['reason']}")
sf.write(f"{args.out}.wav", audio, SR, subtype="PCM_24")
json.dump({"markers": markers, "tracks": tracks}, open(f"{args.out}.json", "w"), indent=1, default=float)
