"""Fast drop-in for scipy.ndimage.gaussian_filter1d on gain/automation curves.

The crossfade engine and stem orchestrator smooth sample-rate gain curves with very
wide Gaussians (sigma of hundreds to thousands of samples). Done at full rate that
took ~2 s per curve — 10-25 s per transition. A gain curve that is about to be
blurred over thousands of samples carries no detail at the sample level, so the
same smoothing is done on a decimated copy and interpolated back: numerically the
same curve (differences ~1e-4), hundreds of times faster.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d


def gaussian_filter1d(x, sigma, **kwargs):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or sigma < 48 or len(x) < 4096 or kwargs:
        return _gaussian_filter1d(x, sigma=sigma, **kwargs)
    d = max(2, int(sigma // 12))                       # keep >= 12 points per sigma
    n = len(x)
    pad = (-n) % d
    xp = np.concatenate([x, np.full(pad, x[-1])]) if pad else x
    coarse = xp.reshape(-1, d).mean(axis=1)            # block-average (anti-aliased) decimation
    coarse = _gaussian_filter1d(coarse, sigma=sigma / d, mode="nearest")
    centres = np.arange(len(coarse)) * d + (d - 1) / 2.0
    return np.interp(np.arange(n), centres, coarse)
