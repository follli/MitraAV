"""Detectors for tone-coded signals hidden in plain sight in audio: DTMF key presses, Morse code (CW),
the SSTV picture header and Bell 202 (1200 baud AFSK, packet radio/APRS).

All work on a mono float signal and return plain dicts with times in seconds. They are deliberately
conservative (a decoded signal needs several consistent symbols), and none prints what DTMF digits were
dialled (a phone number or PIN): the count and timing are reported instead.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, stft

DTMF_ROWS = (697, 770, 852, 941)
DTMF_COLS = (1209, 1336, 1477, 1633)
DTMF_KEYS = ("1", "2", "3", "A", "4", "5", "6", "B", "7", "8", "9", "C", "*", "0", "#", "D")
MORSE = {".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G", "....": "H", "..": "I",
         ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
         "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
         "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5", "-....": "6",
         "--...": "7", "---..": "8", "----.": "9"}
VIS_MODES = {8: "Robot 36 colour", 12: "Robot 72 colour", 40: "Martin M2", 44: "Martin M1", 56: "Scottie S2",
             60: "Scottie S1", 76: "Scottie DX", 93: "PD 50", 95: "PD 120", 97: "PD 180", 98: "PD 240",
             99: "PD 160", 96: "PD 90", 94: "PD 120"}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def dtmf(x: np.ndarray, sr: int) -> list[dict]:
    """Key presses: [{t, dur}] for sequences of at least four dual-tone keys. Digits are not returned."""
    if sr < 4000:
        return []
    nper = int(0.04 * sr)
    f, t, Z = stft(x, sr, nperseg=nper, noverlap=nper // 2, window="hann")
    P = np.abs(Z) ** 2
    idx = lambda hz: int(np.argmin(np.abs(f - hz)))  # noqa: E731
    rows = np.stack([P[idx(h) - 1:idx(h) + 2].sum(axis=0) for h in DTMF_ROWS])
    cols = np.stack([P[idx(h) - 1:idx(h) + 2].sum(axis=0) for h in DTMF_COLS])
    band = (f >= 600) & (f <= 1700)
    total = P[band].sum(axis=0) + 1e-20
    r, c = rows.argmax(axis=0), cols.argmax(axis=0)
    rs, cs = np.sort(rows, axis=0), np.sort(cols, axis=0)
    ok = ((rows.max(axis=0) + cols.max(axis=0)) / total > 0.6)            # the two tones hold the band's energy
    ok &= (rs[-1] > 8 * rs[-2]) & (cs[-1] > 8 * cs[-2])                   # one clear row and one clear column
    ok &= (rows.max(axis=0) / (cols.max(axis=0) + 1e-20) < 8) & (cols.max(axis=0) / (rows.max(axis=0) + 1e-20) < 8)
    key = np.where(ok, r * 4 + c, -1)
    presses, i = [], 0
    while i < len(key):
        if key[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(key) and key[j + 1] == key[i]:
            j += 1
        if j - i + 1 >= 2:
            presses.append(dict(t=float(t[i]), dur=float(t[j] - t[i] + 0.04), key=int(key[i])))
        i = j + 1
    # a real dialling sequence: >= 4 keys, each separated from the next by < 1.5 s
    seqs, cur = [], []
    for p in presses:
        if cur and p["t"] - (cur[-1]["t"] + cur[-1]["dur"]) > 1.5:
            if len(cur) >= 4:
                seqs.append(cur)
            cur = []
        cur.append(p)
    if len(cur) >= 4:
        seqs.append(cur)
    return [dict(p, key=None) for s in seqs for p in s]


def _bimodal_threshold(v: np.ndarray) -> float:
    lo, hi = np.percentile(v, 10), np.percentile(v, 95)
    thr = (lo + hi) / 2
    for _ in range(10):
        a, b = v[v <= thr], v[v > thr]
        if len(a) == 0 or len(b) == 0:
            break
        thr = (a.mean() + b.mean()) / 2
    return float(thr)


def morse(x: np.ndarray, sr: int) -> dict | None:
    """Decode keyed-tone Morse: {text, wpm, tone, t0, t1, valid} or None. Needs >= 6 clean characters."""
    if sr < 4000 or len(x) < sr:
        return None
    hop = 0.005
    nper = int(0.02 * sr)
    f, t, Z = stft(x, sr, nperseg=nper, noverlap=nper - int(hop * sr), window="hann")
    P = np.abs(Z) ** 2
    band = (f >= 300) & (f <= 1500)
    best = None
    for k in np.flatnonzero(band):
        e = P[max(k - 1, 0):k + 2].sum(axis=0)
        if e.max() <= 0:
            continue
        db = 10 * np.log10(e + 1e-20)
        thr = _bimodal_threshold(db)
        if np.percentile(db, 95) - np.percentile(db, 10) < 20:
            continue
        on = db > thr
        score = float(np.std(db) * (0.2 < on.mean() < 0.7))
        if best is None or score > best[0]:
            best = (score, k, on)
    if best is None:
        return None
    _, k, on = best
    runs = _runs(on)
    runs = [(a, b) for a, b in runs if (b - a) * hop >= 0.015]
    if len(runs) < 12:
        return None
    on_d = np.array([(b - a) * hop for a, b in runs])
    unit = float(np.percentile(on_d, 25))
    if not 0.025 <= unit <= 0.25:
        return None
    gaps = np.array([(runs[i + 1][0] - runs[i][1]) * hop for i in range(len(runs) - 1)])
    letters, cur, valid, total = [], "", 0, 0
    for i, (a, b) in enumerate(runs):
        d = (b - a) * hop
        cur += "-" if d > 2 * unit else "."
        gap = gaps[i] if i < len(gaps) else 9 * unit
        if gap > 2 * unit:                                  # end of a letter
            total += 1
            ch = MORSE.get(cur)
            if ch:
                valid += 1
            letters.append(ch or "?")
            cur = ""
            if gap > 5 * unit:
                letters.append(" ")
    text = "".join(letters).strip()
    n_letters = sum(1 for c in text if c != " ")
    if n_letters < 6 or valid < 0.8 * total:
        return None
    dots = on_d[on_d <= 2 * unit]
    unit = float(np.median(dots)) if len(dots) else unit
    return dict(text=text, wpm=1.2 / unit, tone=float(f[k]), t0=float(t[runs[0][0]]), t1=float(t[runs[-1][1] - 1]),
                valid=valid / max(total, 1))


def _dominant(x: np.ndarray, sr: int, lo: float, hi: float, hop: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    nper = int(0.02 * sr)
    f, t, Z = stft(x, sr, nperseg=nper, noverlap=nper - int(hop * sr), window="hann")
    P = np.abs(Z)
    band = (f >= lo) & (f <= hi)
    peak = f[band][P[band].argmax(axis=0)]
    strong = P[band].max(axis=0) > 6 * (np.median(P[band], axis=0) + 1e-12)
    return t, np.where(strong, peak, np.nan)


def sstv(x: np.ndarray, sr: int) -> list[dict]:
    """SSTV calibration headers: [{t, mode}] (1900 Hz leader, a 10 ms break, 1900 Hz leader, VIS code)."""
    if sr < 6000:
        return []
    t, fr = _dominant(x, sr, 1000, 2400)
    leader = np.abs(fr - 1900) <= 60
    runs = [(a, b) for a, b in _runs(leader) if 25 <= b - a <= 45]          # about 300 ms at a 10 ms hop
    out = []
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        if not 1 <= a2 - b1 <= 5:                                           # the short break between the two leaders
            continue
        bits = []
        for i in range(8):                                                  # start bit, then 7 data bits + parity
            c = b2 + 3 + 3 * i + 1
            seg = fr[c:c + 2]
            seg = seg[~np.isnan(seg)]
            bits.append(None if len(seg) == 0 else (1 if abs(np.median(seg) - 1100) < abs(np.median(seg) - 1300) else 0))
        mode = VIS_MODES.get(sum(bit << i for i, bit in enumerate(bits[:7]))) if None not in bits[:7] else None
        out.append(dict(t=float(t[a1]), mode=mode))
    return out


def afsk1200(x: np.ndarray, sr: int) -> list[dict]:
    """Bell 202 AFSK (1200/2200 Hz, 1200 baud) bursts: [{t0, t1, transitions}].

    Each symbol-long window (1/1200 s) is correlated with 1200 Hz and 2200 Hz; a window dominated by
    one of them is a mark or a space, and a burst needs many mark/space alternations in half a second."""
    if sr < 8000:
        return []
    n = max(8, int(round(sr / 1200)))
    t = np.arange(n) / sr
    kern = [np.exp(-2j * np.pi * f * t) for f in (1200.0, 2200.0)]
    energy = np.convolve(x * x, np.ones(n), "valid") + 1e-12
    corr = [np.abs(np.convolve(x, k[::-1].conj(), "valid")) ** 2 / (n * energy) for k in kern]
    # energy-normalised correlation: about 0.5 for a clean tone of that frequency, about 1/n for noise
    c1, c2 = corr[0], corr[1]
    pure = np.maximum(c1, c2) > 0.25
    loud = energy > 0.05 * np.percentile(energy, 99)
    state = np.where(pure & loud, np.where(c1 > c2, 1, 2), 0)
    step = max(1, n // 4)
    state = state[::step]
    dt = step / sr
    win = int(0.5 / dt)
    out = []
    for a in range(0, len(state) - win, win // 2):
        seg = state[a:a + win]
        if (seg > 0).mean() < 0.8:
            continue
        s = seg[seg > 0]
        trans = int((np.diff(s) != 0).sum())
        if (s == 1).mean() > 0.2 and (s == 2).mean() > 0.2 and trans >= 40:
            out.append(dict(t0=a * dt, t1=(a + win) * dt, transitions=trans))
    merged: list[dict] = []
    for o in out:
        if merged and o["t0"] <= merged[-1]["t1"] + 0.05:
            merged[-1]["t1"] = o["t1"]
            merged[-1]["transitions"] += o["transitions"]
        else:
            merged.append(dict(o))
    return [m for m in merged if m["t1"] - m["t0"] >= 0.4]


def downsample(x: np.ndarray, sr: int, target: int = 16000) -> tuple[np.ndarray, int]:
    if sr > target:
        g = np.gcd(sr, target)
        return resample_poly(x, target // g, sr // g), target
    return x, sr
