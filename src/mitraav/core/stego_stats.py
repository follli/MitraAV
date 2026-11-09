"""Statistical steganalysis of decoded samples: what LSB replacement does to a signal.

These tests suit **lossless** carriers (PNG, BMP, TIFF, WAV/FLAC/PCM, lossless video). In a lossy file
(JPEG, MP3, H.264) the least significant bits are rewritten by the codec, so a result there says
nothing, and the probes skip it. A test that finds nothing does not prove nothing is hidden: methods
such as LSB matching, steghide and other adaptive schemes are built to defeat exactly these tests.

* `chi_square_prefix`: Westfeld and Pfitzmann's chi-square attack on pairs of values (2k, 2k+1),
  evaluated on growing prefixes of the data. Replacing LSBs with random bits equalises each pair, so
  the "probability of embedding" climbs towards 1 for the part of the data that carries a message.
* `rs_estimate`: Fridrich, Goljan and Du's RS analysis, an estimate of the share of samples carrying
  LSB-replacement data, from how flipping LSBs changes local smoothness.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

MIN_SAMPLES = 20_000


def pair_chi_square(values: np.ndarray, levels: int = 256) -> tuple[float, int]:
    """(chi-square statistic, degrees of freedom) that the counts of each value pair (2k, 2k+1) are
    equal. Pairs with fewer than 4 expected counts are dropped (the usual rule)."""
    h = np.bincount(values.ravel(), minlength=levels)[:levels].astype(float)
    even, odd = h[0::2], h[1::2]
    expected = (even + odd) / 2
    keep = expected >= 4
    if keep.sum() < 3:
        return 0.0, 0
    stat = float(((even[keep] - expected[keep]) ** 2 / expected[keep]).sum())
    return stat, int(keep.sum()) - 1


def embedding_probability(values: np.ndarray, levels: int = 256) -> float:
    stat, df = pair_chi_square(values, levels)
    return float(1.0 - chi2.cdf(stat, df)) if df > 0 else 0.0


def chi_square_prefix(values: np.ndarray, steps: int = 10, levels: int = 256, with_control: bool = False):
    """[(fraction of the data tested, embedding probability)] over growing prefixes. A message that
    fills the first part of the data shows as probability near 1 up to its length, then a fall.

    With `with_control=True` also returns, for each prefix length, the probability for an equally long
    stretch taken from the END of the data (assumed untouched). If that control also looks "embedded"
    the test has no power at that size (a quiet or flat signal), and the prefix result means nothing."""
    v = np.asarray(values).ravel()
    out, ctl = [], []
    for i in range(1, steps + 1):
        n = len(v) * i // steps
        if n >= MIN_SAMPLES:
            out.append((i / steps, embedding_probability(v[:n], levels)))
            ctl.append(embedding_probability(v[len(v) - n:], levels) if n <= len(v) // 2 else float("nan"))
    return (out, ctl) if with_control else out


def _smoothness(groups: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(groups.astype(np.int32), axis=1)).sum(axis=1)


def _flip(groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the flipping F1 (swap 0<->1, 2<->3, ...) where mask is +1 and F-1 (swap -1<->0, 1<->2, ...)
    where mask is -1; mask 0 leaves the sample alone."""
    g = groups.astype(np.int32)
    out = g.copy()
    f1 = g ^ 1
    fm1 = ((g + 1) ^ 1) - 1
    out = np.where(mask == 1, f1, out)
    out = np.where(mask == -1, fm1, out)
    return np.clip(out, 0, 255)


def _rs_counts(groups: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    base = _smoothness(groups)
    new = _smoothness(_flip(groups, mask))
    n = len(groups)
    return float((new > base).sum()) / n, float((new < base).sum()) / n


def rs_estimate(channel: np.ndarray, group: int = 4, max_groups: int = 2_000_000) -> float | None:
    """Estimated fraction of samples carrying LSB-replacement data (0 = none, 1 = every LSB), or None
    when the data is unsuitable (too small or the equation has no solution). `channel` is a 2-D or
    1-D array of 8-bit samples."""
    data = np.asarray(channel).ravel().astype(np.int32)
    n = (len(data) // group) * group
    if n < MIN_SAMPLES:
        return None
    groups = data[:n].reshape(-1, group)
    if len(groups) > max_groups:
        groups = groups[np.linspace(0, len(groups) - 1, max_groups).astype(int)]
    mask = np.array([0, 1, 1, 0])[:group] if group == 4 else np.array([0, 1] * (group // 2))
    neg = -mask
    r_m, s_m = _rs_counts(groups, mask)
    r_n, s_n = _rs_counts(groups, neg)
    flipped = groups ^ 1                                  # all LSBs flipped
    r_m1, s_m1 = _rs_counts(flipped, mask)
    r_n1, s_n1 = _rs_counts(flipped, neg)
    d0, d1 = r_m - s_m, r_m1 - s_m1
    dn0, dn1 = r_n - s_n, r_n1 - s_n1
    a = 2 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3 * d0
    c = d0 - dn0
    if abs(a) < 1e-12:
        x = -c / b if abs(b) > 1e-12 else None
        roots = [] if x is None else [x]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        roots = [(-b + disc ** 0.5) / (2 * a), (-b - disc ** 0.5) / (2 * a)]
    ps = [x / (x - 0.5) for x in roots if abs(x - 0.5) > 1e-9]
    ps = [p for p in ps if -0.2 <= p <= 1.2]
    if not ps:
        return None
    return float(min(ps, key=abs))                        # the smaller root, as in the paper


def lsb_plane_correlation(channel: np.ndarray) -> float:
    """Correlation between horizontally adjacent LSBs: natural, smooth images keep some, a plane
    filled with random bits has none. A weak supporting clue only."""
    a = (np.asarray(channel) & 1).astype(np.float64)
    if a.ndim != 2 or a.shape[1] < 8:
        return 0.0
    x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def pcm_prefix_values(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """16-bit samples reduced to the values the attack uses: digital-silence runs removed, and only the
    widest central range that still averages 300 counts per value (the pairs need large counts).
    Returns (values shifted to start at 0, number of levels)."""
    s = np.asarray(samples).ravel().astype(np.int64)
    if len(s) == 0:
        return s, 64
    quiet = np.abs(s) <= 1                                   # digital silence would unbalance the pairs (0, 1)
    if quiet.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
        drop = np.zeros(len(s), bool)
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a >= 2048:
                drop[a:b] = True
        s = s[~drop]
    sub = np.abs(s[:: max(1, len(s) // 400_000)])
    scale = len(s) / max(len(sub), 1)
    span = 32
    for cand in (1024, 512, 256, 128, 64, 32):
        if (sub < cand).sum() * scale / (2 * cand) >= 300:
            span = cand
            break
    part = s[(s >= -span) & (s < span)] + span
    return part, 2 * span


def pcm_chi_square_prefix(samples: np.ndarray, steps: int = 10, with_control: bool = False):
    """The chi-square attack for 16-bit PCM, on growing prefixes of the recording."""
    values, levels = pcm_prefix_values(samples)
    return chi_square_prefix(values, steps, levels, with_control)


def silence_lsb_activity(samples: np.ndarray, sr: int, min_run: float = 0.25) -> tuple[float, float]:
    """(seconds of 'near-silence', share of those samples that are not exactly zero). Near-silence is a
    stretch where every sample is 0 or +-1. Exact zeros mean digital silence; a random mix of 0 and +-1
    is either dither or data hidden in the lowest bit, which a hidden-data tool leaves in otherwise
    empty passages."""
    s = np.asarray(samples).ravel()
    quiet = np.abs(s.astype(np.int32)) <= 1
    if not quiet.any():
        return 0.0, 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]
    runs = [(a, b) for a, b in zip(starts, ends) if (b - a) >= min_run * sr]
    if not runs:
        return 0.0, 0.0
    total = sum(b - a for a, b in runs)
    nonzero = sum(int((s[a:b] != 0).sum()) for a, b in runs)
    return total / sr, nonzero / total


def prefix_verdict(prefix: list[tuple[float, float]], hi: float = 0.9, lo: float = 0.1,
                   control: list[float] | None = None) -> tuple[str, float] | None:
    """Read a chi-square prefix curve. ("partial", fraction) when the early part looks like random bits
    and the rest does not (a message filling the start of the data); ("uniform", 1.0) when every prefix
    looks random (noise or dither, or a message that fills everything: indistinguishable); else None."""
    if len(prefix) < 4:
        return None
    ps = [p for _, p in prefix]
    if ps[0] >= hi and ps[-1] <= lo:
        edge = next((f for f, p in prefix if p < 0.5), prefix[-1][0])
        if control is not None:                          # power check at the size of the claimed message
            sizes = [f for f, _ in prefix]
            k = max((i for i, f in enumerate(sizes) if f <= min(edge, 0.5) + 1e-9), default=0)
            c = control[k]
            if not (c == c and c <= 0.3):                # an equally long clean stretch must look natural
                return None
        return "partial", float(edge)
    if min(ps) >= hi:
        return "uniform", 1.0
    return None
