"""Matched-filter Doppler detector: find faint, noisy pass-bys that ridge tracking loses.

The ridge tracker follows one narrow line through the spectrogram, and a burst of wind or traffic noise
cuts that line apart. This detector instead *integrates*: for every candidate pass (speed, closest
distance, time of closest approach, base pitch) it adds up the spectrogram along the whole predicted
curve and along its harmonics, so a faint line that is invisible frame by frame still adds up.

Noise has no preferred direction but an aircraft only ever glides *down* (approach then recession), so the
same scan is repeated with the curve mirrored upward. A pass shows as a downward score well above the
best upward score, and that ratio, not an absolute threshold, is the evidence. The best candidates are
then refined: a ridge is followed along the predicted curve, a Doppler model is fitted to it, and its
numbers are reported. Speed is usually within about 30% in tests; distance is rougher.
"""
from __future__ import annotations

import numpy as np

from .spectro import compute

MF_STEP = 1.01
MF_LN = float(np.log(MF_STEP))
MF_FMIN, MF_FMAX = 50.0, 1600.0
MF_BETAS = (0.04, 0.07, 0.11, 0.17, 0.26, 0.4)           # speed / speed of sound
MF_TAUS = (1.5, 3.0, 6.0, 12.0, 24.0, 48.0)              # closest-approach distance / speed (s)
MF_MIN_RATIO = 1.6                                       # downward score / best upward score
MF_F0_LO, MF_F0_HI = 80.0, 500.0                            # base pitch range of the harmonic comb (Hz)


def zmap(x: np.ndarray, sr: int, nper_s: float = 0.35):
    """Log-frequency map of how far each cell stands above its frame and frequency background."""
    nper = int(2 ** round(np.log2(sr * nper_s)))
    f, t, db = compute(x, sr, nper)
    sel = (f >= MF_FMIN * 0.8) & (f <= MF_FMAX)
    f, db = f[sel], db[sel]
    if len(f) < 8:
        return None
    db = db - np.median(db, axis=0, keepdims=True)      # broadband bursts raise every bin of a frame
    db = db - np.median(db, axis=1, keepdims=True)      # lines that are always present (hum, machinery)
    mad = float(np.median(np.abs(db))) * 1.4826 + 1e-9
    z = np.clip(db / mad, -2, 12)
    yf = MF_FMIN * MF_STEP ** np.arange(int(np.log(MF_FMAX / MF_FMIN) / MF_LN))
    return yf, t, np.stack([np.interp(yf, f, z[:, j]) for j in range(z.shape[1])], axis=1)


def harmonic_sum(z: np.ndarray, k_max: int = 3) -> np.ndarray:
    """Cell i collects the energy at i, 2i, 3i (in frequency): a harmonic comb stacks faint lines."""
    h = np.zeros_like(z)
    for k in range(1, k_max + 1):
        off = int(round(np.log(k) / MF_LN))
        if off < z.shape[0]:
            h[:z.shape[0] - off] += z[off:]
    return h / np.sqrt(k_max)


def scan(h: np.ndarray, t: np.ndarray, sign: int = 1, step_t0: float = 1.0,
         margin: float = 10.0) -> list[tuple[float, float, float, float, int]]:
    """[(score, beta, tau, t0, base pitch bin)] for every (beta, tau, t0): the best base pitch of each."""
    dt = float(t[1] - t[0])
    cols = np.arange(0, h.shape[1], max(1, int(round(0.35 / dt))))
    tc = t[cols]
    lo = int(np.log(MF_F0_LO / MF_FMIN) / MF_LN)
    n_f0 = int(np.log(MF_F0_HI / MF_F0_LO) / MF_LN)
    base = np.arange(n_f0)[:, None] + lo
    margin = min(margin, 0.2 * float(tc[-1] - tc[0]))
    out = []
    for beta in MF_BETAS:
        for tau in MF_TAUS:
            for t0 in np.arange(tc[0] + margin, tc[-1] - margin, step_t0):
                u = (tc - t0) / tau
                g = 1 / (1 + sign * beta * u / np.sqrt(1 + u * u))
                idx = base + np.round(np.log(g) / MF_LN).astype(int)[None, :]
                ok = (idx >= 0) & (idx < h.shape[0])
                vals = np.where(ok, h[np.clip(idx, 0, h.shape[0] - 1), cols[None, :]], 0.0)
                sc = vals.sum(axis=1) / np.sqrt(len(tc))
                i = int(np.argmax(sc))
                out.append((float(sc[i]), beta, tau, float(t0), i + lo))
    return out


def refine(yf, t, z, cand, fit_doppler, k_max: int = 5, band: float = 0.05, zmin: float = 2.5) -> dict | None:
    """Follow a ridge along the candidate curve (best harmonic), fit the Doppler model, return its numbers."""
    _, beta, tau, t0, i0 = cand
    f0 = yf[min(i0, len(yf) - 1)]
    u = (t - t0) / tau
    g = 1 / (1 + beta * u / np.sqrt(1 + u * u))
    best = None
    for k in range(1, k_max + 1):
        yi = (np.log(k * f0 * g) - np.log(yf[0])) / MF_LN
        if yi.min() < 3 or yi.max() > len(yf) - 4:
            continue
        e = float(z[np.clip(np.round(yi).astype(int), 0, len(yf) - 1), np.arange(len(t))].sum())
        if best is None or e > best[0]:
            best = (e, k)
    if best is None:
        return None
    k = best[1]
    pred = np.log(k * f0 * g)
    w = int(round(band / MF_LN))
    pts = []
    for j in range(len(t)):
        c = int(round((pred[j] - np.log(yf[0])) / MF_LN))
        a, b = max(0, c - w), min(len(yf), c + w + 1)
        if b - a < 3:
            continue
        seg = z[a:b, j]
        m = int(np.argmax(seg))
        if seg[m] >= zmin:
            pts.append((t[j], yf[a + m], seg[m]))
    if len(pts) < 20:
        return None
    tt, ff, zz = map(np.array, zip(*pts))
    if tt[-1] - tt[0] < 10:
        return None
    fit = fit_doppler(tt, ff)
    if not fit:
        return None
    p, r2 = fit
    return dict(t0=float(p[2]), beta=float(p[1]), tau=float(p[3]), r2=float(r2), n=len(tt), harmonic=k,
                t_start=float(tt[0]), t_end=float(tt[-1]), hi=float(np.median(ff[:5])), lo=float(np.median(ff[-5:])),
                zmean=float(zz.mean()))


def detect(x: np.ndarray, sr: int, fit_doppler, offset: float = 0.0, max_events: int = 3) -> list[dict]:
    """Candidate passes in one chunk of audio, best first. Each has `ratio` (down vs up evidence) and the
    refined Doppler numbers; only candidates with ratio >= MF_MIN_RATIO and a successful fit are returned."""
    zm = zmap(x, sr)
    if zm is None or len(zm[1]) < 40:
        return []
    yf, t, z = zm
    h = harmonic_sum(z)
    long_chunk = (t[-1] - t[0]) > 150
    down = scan(h, t, +1, step_t0=2.0 if long_chunk else 1.0)
    up = scan(h, t, -1, step_t0=2.0 if long_chunk else 1.0)
    if not down or not up:
        return []
    best_up = max(c[0] for c in up)
    out, taken = [], []
    for cand in sorted(down, reverse=True):
        if len(out) >= max_events or cand[0] / best_up < MF_MIN_RATIO:
            break
        if any(abs(cand[3] - t1) < 20.0 for t1 in taken):
            continue
        taken.append(cand[3])
        r = refine(yf, t, z, cand, fit_doppler)
        if not r:
            continue
        r.update(ratio=float(cand[0] / best_up), t0_template=float(cand[3]))
        for key in ("t0", "t_start", "t_end", "t0_template"):
            r[key] += offset
        out.append(r)
    return out
