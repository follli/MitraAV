from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch

from ..core.audio import AudioError, load_audio
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

# 300 Hz is skipped: it is a harmonic of both grids.
GRIDS = {50: (50, 100, 150, 200, 250), 60: (60, 120, 180, 240)}
HIT_DB = 9.0   # a peak this far above its neighbours counts as a hit
FUND_DB = 12.0


def _ratio_db(f, p, target):
    dist = np.abs(f - target)
    near = dist <= 0.6
    ring = (dist >= 2) & (dist <= 8)
    if not near.any() or ring.sum() < 4:
        return None
    i = int(np.argmax(np.where(near, p, -np.inf)))
    ref = max(float(np.median(p[ring])), 1e-20)
    return 10 * np.log10(max(float(p[i]), 1e-20) / ref), float(f[i])


def analyse(x: np.ndarray, sr: int) -> dict[int, list[tuple[int, float, float]]]:
    nper = int(min(len(x), sr * 8))
    f, p = welch(x, fs=sr, nperseg=nper, window="hann",
                 noverlap=nper // 2 if len(x) > nper else 0)
    out = {}
    for grid, harmonics in GRIDS.items():
        hits = []
        for h in harmonics:
            if h + 8 >= sr / 2:
                continue
            r = _ratio_db(f, p, h)
            if r and r[0] >= HIT_DB:
                hits.append((h, r[0], r[1]))
        out[grid] = hits
    return out


def enf_track(x: np.ndarray, sr: int, grid: int, harmonics: list[int], win: float = 4.0,
              hop: float = 2.0) -> np.ndarray:
    """Estimated supply frequency (Hz) per window: each detected harmonic h is located by a
    zero-padded FFT with parabolic interpolation, divided by h (so a harmonic gives h times the
    resolution), and the harmonics are combined with a median. NaN where nothing is found."""
    n, step = int(win * sr), int(hop * sr)
    if len(x) < n + step:
        return np.array([])
    w = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * 4)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    out = []
    for a in range(0, len(x) - n, step):
        mag = np.abs(np.fft.rfft(x[a:a + n] * w, nfft))
        ests = []
        for h in harmonics:
            nominal = h
            band = np.flatnonzero(np.abs(freqs - nominal) <= 0.5 * (nominal / grid))
            if band.size < 3:
                continue
            i = band[int(np.argmax(mag[band]))]
            if 0 < i < len(mag) - 1 and mag[i] > 0:
                al, be, ga = np.log(mag[i - 1] + 1e-12), np.log(mag[i] + 1e-12), np.log(mag[i + 1] + 1e-12)
                d = 0.5 * (al - ga) / (al - 2 * be + ga) if (al - 2 * be + ga) != 0 else 0.0
                ests.append((freqs[i] + d * (freqs[1] - freqs[0])) / (h / grid))
        out.append(float(np.median(ests)) if ests else np.nan)
    return np.array(out)


def _detected(grid: int, hits) -> bool:
    return len(hits) >= 2 or any(h == grid and db >= FUND_DB for h, db, _ in hits)


class HumProbe(Probe):
    name = "mains_hum"
    group = "Setting and power"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 20.0

    def run(self, path: Path) -> Finding:
        try:
            x, sr = load_audio(path)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        dur = len(x) / sr
        if dur < 2 or sr < 600:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Need at least 2 s of audio sampled at 600 Hz or more")

        res = analyse(x, sr)
        det = {g: h for g, h in res.items() if _detected(g, h)}
        if not det:
            return Finding(
                probe=self.name, detected=False,
                summary="No 50/60 Hz hum above the noise floor. Phone mics, codecs and "
                        "battery-powered recorders often remove it, so absence says little.")

        total = {g: sum(db for _, db, _ in h) for g, h in det.items()}
        best = max(total, key=total.get)
        others = [v for g, v in total.items() if g != best]
        ambiguous = bool(others) and max(others) > 0.67 * total[best]
        hits = det[best]
        top = max(db for _, db, _ in hits)
        strength = min(1.0, max(0.2, (top - 6) / 24 + 0.1 * (len(hits) - 1)))
        if ambiguous:
            strength *= 0.5

        peaks = ", ".join(str(h) for h, _, _ in hits)
        notes = [f"Peaks at {peaks} Hz are consistent with a **{best} Hz mains supply** "
                 f"(hum can also come from nearby equipment)"]
        fund = [fr for h, _, fr in hits if h == best]
        if fund:
            notes.append(f"measured {fund[0]:.2f} Hz")
        if ambiguous:
            notes.append("50 and 60 Hz grids both show peaks, so treat as ambiguous")
        if dur >= 20:
            tr = enf_track(x, sr, best, [h for h, _, _ in hits])
            tr = tr[~np.isnan(tr)]
            if len(tr) >= 5:
                notes.append(f"the frequency track runs from {tr.min():.3f} to {tr.max():.3f} Hz over "
                             f"{dur:.0f} s (a grid frequency wanders slowly like this; a perfectly "
                             "steady tone points to equipment or an inverter) and can be compared with "
                             "published grid-frequency logs")
        if dur < 60:
            notes.append(f"only {dur:.0f} s analysed; ENF log matching works best with minutes")

        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.5,
            constrains=["region", "date"], effort=2,
            follow_ups=follow_ups("audio", "enf"), summary="; ".join(notes) + ".")
