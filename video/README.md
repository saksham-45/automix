# automix × visualizzer video pipeline

Renders an automix mix with [visualizzer](https://github.com/saksham-45/visualizzer)
as the picture, plus an automix HUD (now playing, the transition move, a deck
crossfader and the set timeline). The output is 1080p60 H.264 with 320 kbps AAC.

Rendering is **offline and frame-accurate**. The page runs on a virtual clock:
`performance.now`, `Date.now`, `requestAnimationFrame` and timers are replaced, and
`Math.random` is seeded. For every video frame, the renderer:

1. injects exactly what a Web Audio `AnalyserNode` (fftSize 2048, smoothing 0.8,
   −100…−30 dB) would have returned at that moment;
2. steps the clock 1/60 s;
3. takes a screenshot.

A slow GPU makes the render take longer but never drops or smears a frame, and the
same input gives the same video every time.

## Steps

```bash
# 0. once
cd video && npm install            # playwright-core; uses your installed Chrome
# serve a visualizzer checkout
(cd ../../visualizzer && python3 -m http.server 8765) &

# 1. the mix: quick-mix mode, 16-bar transitions, stems on
python video/make_quickmix.py "Kanye West - I Wonder.wav" "Kanye West - Champion.wav" ... --out qm
# 2. trim only the two ends for the video (nothing inside a song is cut)
python video/trim_ends.py qm                       # -> qm_video.wav / qm_video.json
# 3. per-frame analyser data. Metal modes get a hotter input, like a mic in a room.
python video/features.py qm_video.wav frames.bin 60 -38
python video/features.py qm_video.wav frames_metal.bin 60 -15
# 4. render
node video/render.mjs frames.bin qm_video.json qm_video.wav out.mp4 0 '' frames_metal.bin \
     "psychedelic rap" "music © the respective artists"
```

`VIZ_URL` points the renderer at a different visualizer URL. The visual mode changes
at the midpoint of each transition and follows the `ROTATION` list in `render.mjs`:
Chrome Morph, Neon Vortex, Mercury Orbs, Electric Storm, Liquid Metal, Hypnotic
Spiral, Metallic Nebula. On an M1 Pro it renders at about 17 fps, so a 4-minute
video takes about 15 minutes.
