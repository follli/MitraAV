"""Church bells: find strikes whose partials follow the tuned-bell pattern and report each
bell's pitch, so it can be looked up in a tower database (Dove's Guide for UK rings).

Tuned bell partials relative to the prime: hum 0.5, prime 1, tierce 1.2, quint 1.5, nominal 2.
The nominal is the pitch a bell is conventionally named by.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import spectrogram

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding
from ..core.references import follow_ups

RATIOS = (0.5, 1.0, 1.2, 1.5, 2.0)      # hum, prime, tierce, quint, nominal
TOL = 0.018                             # relative frequency tolerance per partial
MIN_PEAK_DB = 12.0
PRIME_MIN, PRIME_MAX = 80.0, 1100.0     # church bells; higher 'primes' are chance alignments
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(f: float) -> str:
    m = 69 + 12 * np.log2(f / 440.0)
    i = int(round(m))
    cents = (m - i) * 100
    return f"{NOTES[i % 12]}{i // 12 - 1} {cents:+.0f} cents"


def onsets(x: np.ndarray, sr: int) -> list[float]:
    """Strike times (s): sudden broadband energy rises, at least 0.8 s apart."""
    n = int(sr * 0.05)
    f, t, s = spectrogram(x, fs=sr, nperseg=n, noverlap=0, mode="psd")
    m = (f >= 100) & (f <= 6000)
    env = 10 * np.log10(s[m].sum(axis=0) + 1e-20)
    out: list[float] = []
    for i in range(8, len(env) - 2):
        base = float(np.median(env[i - 8:i - 2]))
        if env[i] - base >= 8.0 and env[i] == env[max(0, i - 3):i + 4].max():
            if not out or t[i] - out[-1] >= 0.8:
                out.append(float(t[i]))
    return out


def spectrum_peaks(seg: np.ndarray, sr: int):
    w = np.hanning(len(seg))
    nfft = int(2 ** np.ceil(np.log2(len(seg) * 4)))
    mag = np.abs(np.fft.rfft(seg * w, nfft))
    db = 20 * np.log10(mag + 1e-12)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    keep = (f >= 60) & (f <= 6000)
    f, db = f[keep], db[keep]
    df = f[1] - f[0]
    floor = uniform_filter1d(db, int(300 / df), mode="nearest")
    is_pk = (db == maximum_filter1d(db, int(max(3, 8 / df)))) & (db - floor >= MIN_PEAK_DB)
    idx = np.flatnonzero(is_pk)
    # parabolic refinement
    freqs = []
    for i in idx:
        if 0 < i < len(db) - 1:
            a, b, c = db[i - 1], db[i], db[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            freqs.append(float(f[i] + d * df))
        else:
            freqs.append(float(f[i]))
    return np.array(freqs), (db - floor)[idx]


def match_bell(freqs: np.ndarray, prom: np.ndarray):
    """Best prime frequency whose partials appear at the tuned-bell ratios, or None."""
    best = None
    for p in freqs:
        if not PRIME_MIN <= p <= PRIME_MAX:
            continue
        found, est = [], []
        for r in RATIOS:
            c = np.flatnonzero(np.abs(freqs - r * p) <= TOL * r * p)
            if c.size:
                j = c[np.argmin(np.abs(freqs[c] - r * p))]
                found.append(r)
                est.append(freqs[j] / r)
        # the minor third (tierce) is what separates a bell from any harmonic sound such as a voice
        core = {1.0, 1.2, 2.0} <= set(found)
        if len(found) >= 4 and core:
            f0 = float(np.mean(est))
            cand = (len(found), -float(np.std(est) / f0), f0, found)
            if best is None or cand[:2] > best[:2]:
                best = cand
    return best


def analyse(x: np.ndarray, sr: int, offset: float) -> list[dict]:
    strikes = []
    for t in onsets(x, sr):
        a, b = int((t + 0.25) * sr), int((t + 2.0) * sr)
        if b > len(x) or b - a < sr:
            continue
        freqs, prom = spectrum_peaks(x[a:b], sr)
        if len(freqs) < 4:
            continue
        m = match_bell(freqs, prom)
        if m:
            n, _, f0, found = m
            strikes.append(dict(t=offset + t, prime=f0, nominal=2 * f0, partials=n))
    return strikes


def cluster(strikes: list[dict]) -> list[dict]:
    """Group strikes by pitch (within 2%), i.e. one entry per distinct bell."""
    out: list[dict] = []
    for s in sorted(strikes, key=lambda s: s["nominal"]):
        if out and abs(np.log(s["nominal"] / out[-1]["nominal"])) < 0.02:
            c = out[-1]
            c["n"] += 1
            c["nominal"] = float(np.mean([c["nominal"], s["nominal"]]))
            c["times"].append(s["t"])
            c["partials"] = max(c["partials"], s["partials"])
        else:
            out.append(dict(nominal=s["nominal"], n=1, times=[s["t"]], partials=s["partials"]))
    return out


class ChurchBellsProbe:
    """Part of the combined tones_chimes probe (see tones.py); not a probe on its own."""

    name = "church_bells"

    def run(self, path: Path) -> Finding:
        strikes: list[dict] = []
        n = 0
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 5 or sr < 8000:
                    continue
                n += 1
                strikes += analyse(x, sr, off)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 5 s at 8 kHz or more")
        uniq: list[dict] = []
        for s in sorted(strikes, key=lambda s: s["t"]):       # overlapping chunks repeat strikes
            if not uniq or s["t"] - uniq[-1]["t"] > 0.5:
                uniq.append(s)
        bells = [b for b in cluster(uniq) if b["n"] >= 2 or b["partials"] == 5]
        if not bells:
            return Finding(
                probe=self.name, detected=False,
                summary="No strike with the tuned-bell partial pattern (hum, prime, tierce, quint, "
                        "nominal). Distance, wind and reverberation can blur bell partials.")
        total = sum(b["n"] for b in bells)
        strength = min(1.0, 0.3 + 0.1 * min(total, 6) + 0.05 * (max(b["partials"] for b in bells) - 4)
                       + 0.05 * min(len(bells) - 1, 4))
        notes = [f"{total} strike(s) show the tuned-bell partial pattern, consistent with "
                 f"{len(bells)} distinct bell{'s' if len(bells) != 1 else ''}"]
        for b in bells[:8]:
            notes.append(f"nominal about **{b['nominal']:.0f} Hz ({note_name(b['nominal'])})**, "
                         f"{b['n']} strike{'s' if b['n'] != 1 else ''} near {b['times'][0]:.0f} s")
        if len(bells) >= 3:
            notes.append("the set of pitches can be compared with the tower's ring (tenor note "
                         "and scale), which narrows candidates a lot")
        notes.append("pitch estimates are rough (reverberation, distance and tuning of old bells "
                     "shift partials): search with a tolerance of a few percent, and compare "
                     "nominal, not strike note")
        regions = [dict(label=f"bell ~{b['nominal']:.0f} Hz near {b['times'][0]:.0f} s",
                        t0=max(0.0, b["times"][0] - 1), t1=b["times"][0] + 4,
                        f0=0.8 * b["nominal"] / 4, f1=1.6 * b["nominal"])
                   for b in bells[:8]]
        return Finding(probe=self.name, detected=True, strength=strength,
                       meta={"context": {"bells": {"strikes": [round(s["t"], 2) for s in uniq], "distinct": len(bells)}}},
                       narrows=0.4 if len(bells) >= 3 else 0.25,
                       constrains=["region", "place"], effort=2, regions=regions,
                       follow_ups=follow_ups("audio", "bells"),
                       summary="; ".join(notes) + ".")
