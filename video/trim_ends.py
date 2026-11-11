"""Trim ONLY the two ends of a quick mix for the video (nothing inside a song is cut):
start 4 bars before the first transition, end 8 bars after the last one.

usage: python video/trim_ends.py qm   ->  qm_video.wav + qm_video.json
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import club_mixer as cm  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "qm"
SR = 44100
y, _ = sf.read(f"{name}.wav", dtype="float32")
D = json.load(open(f"{name}.json"))
M = D["markers"]
tr = [m for m in M if m["type"] == "transition"]
first, last = tr[0], tr[-1]
# start: the downbeat 4 bars before the first transition (grid of the first song)
g0 = cm.build_phrase_grid(y[: int(first["start_sec"] * SR)], SR)
db = g0.downbeats[g0.downbeats < first["start_sec"] - 0.01]
start = float(db[-4]) if len(db) >= 4 else 0.0
# end: 8 bars after the last transition, on the last song's grid
g1 = cm.build_phrase_grid(y[int(last["end_sec"] * SR):], SR)
end = last["end_sec"] + float(g1.downbeats[min(8, len(g1.downbeats) - 1)])
out = y[int(start * SR):int(end * SR)].copy()
fi, fo = int(1.5 * SR), int(6.0 * SR)
out[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None] ** 2
out[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None] ** 1.5
nm = []
for m in M:
    a, b = max(m["start_sec"], start) - start, min(m["end_sec"], end) - start
    if b <= a:
        continue
    mm = dict(m)
    mm["start_sec"], mm["end_sec"] = a, b
    nm.append(mm)
D["markers"] = nm
sf.write(f"{name}_video.wav", out, SR, subtype="PCM_24")
json.dump(D, open(f"{name}_video.json", "w"), indent=1, default=float)
print(f"video audio {start:.1f}s -> {end:.1f}s = {end - start:.1f}s ({(end - start) / 60:.2f} min)")
