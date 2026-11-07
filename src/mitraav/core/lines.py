"""Stable narrow tonal lines: horizontal ridges in a spectrogram (complements tracks.py)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_closing,
    find_objects,
    label,
    maximum_filter1d,
    median_filter,
)

from .spectro import compute


@dataclass
class Line:
    f: float          # Hz
    t0: float         # s (file time)
    t1: float
    prom: float       # dB above the local spectral floor
    constant: bool    # present for nearly the whole analysed stretch (hum, machinery)
    level: float = 0.0  # mean level of the line in dB (absolute, unlike prom)


def find_lines(x: np.ndarray, sr: int, offset: float = 0.0, fmin: float = 150.0,
               fmax: float = 5000.0, min_dur: float = 0.35, thr: float = 12.0) -> list[Line]:
    n = int(2 ** round(np.log2(sr * 0.2)))
    if len(x) < n * 4:
        return []
    f, t, db = compute(x, sr, n)
    rows = (f >= fmin) & (f <= min(fmax, 0.45 * sr))
    f, db = f[rows], db[rows]
    if len(f) < 50:
        return []
    # Prominence against the local median (a strong tone must not raise its own floor), and only at
    # true local maxima, so a very strong pure tone is one thin line, not a tall leakage blob.
    pk = db - median_filter(db, size=(41, 1), mode="nearest")
    is_peak = db >= maximum_filter1d(db, size=5, axis=0)
    mask = binary_closing((pk >= thr) & is_peak, structure=np.ones((3, 3), bool))
    lab, k = label(mask, structure=np.ones((3, 3), bool))
    hop = float(t[1] - t[0])
    total = len(x) / sr
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 6 or w * hop < min_dur:
            continue
        m = lab[sl] == i
        if m.sum() < 0.4 * h * w:
            continue
        wt = np.where(m, pk[sl], 0.0)
        fc = float((wt.sum(axis=1) * f[sl[0]]).sum() / max(wt.sum(), 1e-9))
        dur = (sl[1].stop - sl[1].start) * hop
        out.append(Line(f=fc, t0=offset + float(t[sl[1].start]), t1=offset + float(t[sl[1].stop - 1]),
                        prom=float(wt.sum() / m.sum()), constant=bool(total >= 60 and dur >= 0.85 * total),
                        level=float(db[sl][m].mean())))
    return out
