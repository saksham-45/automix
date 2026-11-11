"""Per-video-frame audio data, exactly what a Web Audio AnalyserNode (fftSize 2048,
smoothing 0.8, -100..-30 dB) would hand the visualizer at each animation frame."""
import sys, numpy as np, soundfile as sf
wav, out, fps = sys.argv[1], sys.argv[2], int(sys.argv[3])
gain = 10 ** (float(sys.argv[4]) / 20) if len(sys.argv) > 4 else 1.0
y, sr = sf.read(wav, dtype="float32")
mono = (y.mean(axis=1) if y.ndim == 2 else y) * gain
N, tau = 2048, 0.8
win = np.blackman(N).astype(np.float32)
nframes = int(len(mono) / sr * fps)
mono = np.concatenate([np.zeros(N, np.float32), mono, np.zeros(N, np.float32)])
smooth = np.zeros(N // 2, np.float64)
res = np.empty((nframes, N), np.uint8)
for f in range(nframes):
    end = N + int(round(f / fps * sr))
    seg = mono[end - N:end]
    mag = np.abs(np.fft.rfft(seg * win))[: N // 2] / N
    smooth = tau * smooth + (1 - tau) * mag
    db = 20 * np.log10(np.maximum(smooth, 1e-12))
    res[f, : N // 2] = np.clip(255 * (db + 100) / 70, 0, 255).astype(np.uint8)
    res[f, N // 2:] = np.clip(128 * (1 + seg[-N // 2:]), 0, 255).astype(np.uint8)
res.tofile(out)
print(nframes, "frames", f"{nframes / fps / 60:.1f} min", "mean freq byte", float(res[:, :512].mean()))
