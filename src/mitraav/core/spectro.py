from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram

_POS = [0.0, 0.25, 0.5, 0.75, 1.0]
_R = [0, 87, 188, 249, 252]
_G = [0, 16, 55, 142, 255]
_B = [4, 110, 84, 8, 164]
_x = np.linspace(0, 1, 256)
LUT = np.stack([np.interp(_x, _POS, c) for c in (_R, _G, _B)], axis=1).astype(np.uint8)


def compute(samples: np.ndarray, sr: int, nperseg: int = 4096):
    """Return (freqs, times, dB matrix shaped [freq, time])."""
    nperseg = min(nperseg, len(samples))
    f, t, s = spectrogram(samples, fs=sr, nperseg=nperseg, noverlap=nperseg // 2,
                          window="hann", mode="psd")
    return f, t, (10 * np.log10(s + 1e-12)).astype(np.float32)


def to_rgb(db: np.ndarray, lo_pct: float = 20.0, hi_pct: float = 99.8) -> np.ndarray:
    """dB matrix -> uint8 RGB image, low frequencies at the bottom."""
    lo, hi = np.percentile(db, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    idx = (np.clip((db - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(LUT[idx][::-1])
