from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectro import compute

STEP = 1.005   # log-frequency grid: 0.5% per bin, so a Doppler shift is a pure translation
PEN = 0.4      # dB penalty per bin of movement between frames


@dataclass
class Track:
    t: np.ndarray      # seconds
    f: np.ndarray      # Hz
    prom: np.ndarray   # dB above the frame background
    level: np.ndarray  # raw dB at the track (for the loudness-envelope check)
    span: int          # frames from first to last point, inclusive
    narrow: float = 0.0  # median dB the ridge stands above its neighbouring bins (tonality)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def coverage(self) -> float:
        return len(self.t) / max(self.span, 1)


def pick_nperseg(sr: int) -> int:
    n = 2 ** int(round(np.log2(sr * 0.18)))
    return int(min(8192, max(512, n)))


def _ridge(score: np.ndarray, maxj: int) -> np.ndarray:
    """Best smooth path through score[freq, time] (Viterbi). Returns a bin index per frame."""
    nf, nt = score.shape
    acc = score[:, 0].copy()
    back = np.zeros((nf, nt), dtype=np.int8)
    for j in range(1, nt):
        best = np.full(nf, -np.inf)
        arg = np.zeros(nf, dtype=np.int8)
        for s in range(-maxj, maxj + 1):          # previous bin = i - s
            if s == 0:
                cand = acc.copy()
            else:
                cand = np.full(nf, -np.inf)
                if s > 0:
                    cand[s:] = acc[:nf - s]
                else:
                    cand[:nf + s] = acc[-s:]
            cand -= PEN * abs(s)
            m = cand > best
            best[m], arg[m] = cand[m], s
        acc = best + score[:, j]
        back[:, j] = arg
    path = np.zeros(nt, dtype=int)
    i = int(np.argmax(acc))
    path[-1] = i
    for j in range(nt - 1, 0, -1):
        i = i - int(back[i, j])
        path[j - 1] = i
    return path


def _narrowness(res: np.ndarray, path: np.ndarray, idx: np.ndarray) -> float:
    """Median dB by which the ridge exceeds its neighbours 5-15 bins (2.5-7.5%) either side.
    Tones stand out; broadband bursts and noise do not."""
    nf = res.shape[0]
    out = []
    for j in idx:
        p = int(path[j])
        side = np.r_[res[max(0, p - 15):max(0, p - 4), j], res[p + 5:min(nf, p + 16), j]]
        if side.size:
            out.append(float(res[p, j] - np.median(side)))
    return float(np.median(out)) if out else 0.0


def _merge(tracks: list[Track], hop: float, max_gap_s: float = 2.0, tol: float = 0.02) -> list[Track]:
    """Join duplicates from overlapping segments and continuations across segment edges."""
    tracks = sorted(tracks, key=lambda tr: tr.t[0])
    out: list[Track] = []
    for tr in tracks:
        for k, o in enumerate(out):
            if tr.t[0] > o.t[-1] + max_gap_s:
                continue
            mo, mt = o.t >= tr.t[0] - 1.0, tr.t <= o.t[-1] + 1.0
            a = float(np.median(o.f[mo])) if mo.any() else float(o.f[-1])
            b = float(np.median(tr.f[mt])) if mt.any() else float(tr.f[0])
            if abs(np.log(a / b)) > tol:
                continue
            t = np.concatenate([o.t, tr.t])
            order = np.argsort(t, kind="stable")
            tu, keep = np.unique(t[order], return_index=True)
            pick = order[keep]

            def cat(u, v):
                return np.concatenate([u, v])[pick]

            out[k] = Track(t=tu, f=cat(o.f, tr.f), prom=cat(o.prom, tr.prom),
                           level=cat(o.level, tr.level),
                           span=int(round((tu[-1] - tu[0]) / hop)) + 1,
                           narrow=max(o.narrow, tr.narrow))
            break
        else:
            out.append(tr)
    return out


def trace(x: np.ndarray, sr: int, nperseg: int | None = None, fmin: float = 80.0,
          fmax: float = 5000.0, min_prom_db: float = 9.0, min_dur: float = 3.0,
          max_gap: int = 6, passes: int = 3, seg_s: float = 30.0) -> list[Track]:
    """Follow moving narrow tones. Stationary lines are removed; the clip is cut into
    overlapping segments (a single global path jumps between clutter), each pass masks the last
    ridge, and tracks that continue across segment edges are stitched back together."""
    n = nperseg or pick_nperseg(sr)
    if len(x) < n * 4:
        return []
    fhi = min(fmax, 0.95 * sr / 2)
    if fhi <= fmin * 2:
        return []
    f, t, db = compute(x, sr, n)
    rows = f <= fhi * 1.01
    f, db = f[rows], db[rows]
    grid = fmin * STEP ** np.arange(int(np.log(fhi / fmin) / np.log(STEP)))
    edges = np.clip(np.searchsorted(f, grid), 0, len(f) - 1)
    res = np.maximum.reduceat(db, edges, axis=0)      # max-pool: narrow peaks survive

    rel = res - np.median(res, axis=1, keepdims=True)   # drop anything always present
    rel -= np.median(rel, axis=0, keepdims=True)        # per-frame background
    full_score = np.clip(rel, 0, 30)

    hop = n / 2 / sr
    maxj = int(min(40, max(2, np.ceil(30 * hop))))      # allow ~15%/s of pitch change
    L = max(40, int(seg_s / hop))
    step = max(20, L // 2)
    tracks: list[Track] = []
    for s0 in range(0, max(1, rel.shape[1] - L // 2), step):
        sl = slice(s0, min(s0 + L, rel.shape[1]))
        if sl.stop - sl.start < 20:
            continue
        score = full_score[:, sl].copy()
        r, rs, tt = rel[:, sl], res[:, sl], t[sl]
        cols = np.arange(score.shape[1])
        for _ in range(passes):
            path = _ridge(score, maxj)
            prom, fpk, level = r[path, cols], grid[path], rs[path, cols]
            valid = np.flatnonzero(prom >= min_prom_db)
            runs, cur = [], []
            for i in valid:
                if cur and i - cur[-1] > max_gap:
                    runs.append(cur)
                    cur = []
                cur.append(int(i))
            if cur:
                runs.append(cur)
            for run in runs:
                if len(run) < 8:
                    continue
                idx = np.array(run)
                tracks.append(Track(t=tt[idx], f=fpk[idx], prom=prom[idx], level=level[idx],
                                    span=int(idx[-1] - idx[0] + 1),
                                    narrow=_narrowness(rs, path, idx)))
            for j, p in enumerate(path):                # mask this ridge, look for the next
                score[max(0, p - 10):p + 11, j] = 0
    tracks = [tr for tr in _merge(tracks, hop) if tr.duration >= min_dur and tr.coverage >= 0.5]
    tracks.sort(key=lambda tr: -tr.duration)
    return tracks[:8]


def _trend(tr: Track, end: bool, span_s: float = 5.0) -> tuple[float, float] | None:
    """(ln f at the track's end or start, slope of ln f per second near it), or None if too short."""
    m = (tr.t >= tr.t[-1] - span_s) if end else (tr.t <= tr.t[0] + span_s)
    if m.sum() < 4:
        return None
    t, lf = tr.t[m], np.log(tr.f[m])
    slope = float(np.polyfit(t, lf, 1)[0])
    return float(np.polyval([slope, np.mean(lf) - slope * np.mean(t)], t[-1] if end else t[0])), slope


def stitch(tracks: list[Track], max_gap_s: float = 9.0, tol: float = 0.035) -> list[Track]:
    """Join pieces of one glide that a burst of broadband noise cut apart.

    Two tracks are joined when the second starts after the first ends (within max_gap_s) and the pitch
    change across the gap matches the slope of both ends (a pass-by glide is smooth): the straight
    extrapolation of each across the gap must land within `tol` (a ratio of 3.5%) of the other. The
    joined track keeps only real measurements (the gap is left empty), so downstream checks still see it
    as broken there. Pieces of different harmonics are not joined (their pitch differs by far more)."""
    pool = sorted(tracks, key=lambda tr: tr.t[0])
    changed = True
    while changed:
        changed = False
        best = None
        for i, a in enumerate(pool):
            ta = _trend(a, end=True)
            if ta is None:
                continue
            for j, b in enumerate(pool):
                if i == j or not 0.0 < b.t[0] - a.t[-1] <= max_gap_s:
                    continue
                tb = _trend(b, end=False)
                if tb is None:
                    continue
                gap = float(b.t[0] - a.t[-1])
                (la, sa), (lb, sb) = ta, tb
                s = 0.5 * (sa + sb)
                resid = abs((lb - la) - s * gap)
                forward = abs(la + sa * gap - lb)
                backward = abs(lb - sb * gap - la)
                if max(forward, backward) <= tol + 0.5 * abs(s) * gap * 0.15 and resid <= tol and sa * sb >= -1e-6:
                    if best is None or resid < best[0]:
                        best = (resid, i, j)
        if best:
            _, i, j = best
            a, b = pool[i], pool[j]
            hop = float(np.median(np.diff(np.concatenate([a.t, b.t])))) if len(a.t) + len(b.t) > 3 else 0.2
            merged = Track(t=np.concatenate([a.t, b.t]), f=np.concatenate([a.f, b.f]),
                           prom=np.concatenate([a.prom, b.prom]), level=np.concatenate([a.level, b.level]),
                           span=int(round((b.t[-1] - a.t[0]) / max(hop, 1e-3))) + 1, narrow=max(a.narrow, b.narrow))
            pool = [tr for k, tr in enumerate(pool) if k not in (i, j)] + [merged]
            pool.sort(key=lambda tr: tr.t[0])
            changed = True
    return pool
