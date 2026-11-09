from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch

from ..core.audio import AudioError, load_audio
from ..core.finding import Finding, MediaType
from ..core.geo import grid_hint
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


def comb_search(x: np.ndarray, sr: int) -> tuple[float, list[int]] | None:
    """A mains-like harmonic series whose fundamental is NOT 50 or 60 Hz: (fundamental Hz, harmonics hit).
    Looks at 40-70 Hz in 0.05 Hz steps and needs at least three of the first four harmonics."""
    nper = int(min(len(x), sr * 8))
    f, p = welch(x, fs=sr, nperseg=nper, window="hann", noverlap=nper // 2 if len(x) > nper else 0)
    best = None
    for f0 in np.arange(40.0, 70.01, 0.05):
        if abs(f0 - 50) <= 0.8 or abs(f0 - 60) <= 0.8:
            continue
        hits = []
        for h in (1, 2, 3, 4):
            if h * f0 + 8 >= sr / 2:
                continue
            r = _ratio_db(f, p, h * f0)
            if r and r[0] >= HIT_DB + 3:
                hits.append((h, r[0]))
        if len(hits) >= 3 and (best is None or sum(d for _, d in hits) > best[0]):
            best = (sum(d for _, d in hits), f0, [h for h, _ in hits])
    if best is None:
        return None
    _, f0, hs = best
    # refine with the highest harmonic: its peak position divided by its number
    h = max(hs)
    near = np.abs(f - h * f0) <= 0.6
    return float(f[near][int(np.argmax(p[near]))] / h), hs


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
            shifted = comb_search(x, sr)
            if shifted:
                f0, hs = shifted
                return Finding(
                    probe=self.name, detected=True, strength=0.3, narrows=0.2, constrains=["origin"], effort=2,
                    follow_ups=follow_ups("audio", "enf"),
                    summary=f"a mains-like hum series with its fundamental at **{f0:.1f} Hz** (harmonics {', '.join(str(h) for h in hs)}) "
                            f"does not match 50 or 60 Hz: consistent with 50 Hz hum in audio sped up by about "
                            f"{(f0 / 50 - 1) * 100:+.1f}%, or 60 Hz hum changed by {(f0 / 60 - 1) * 100:+.1f}%, or with hum from "
                            "equipment on another supply. A changed speed also shifts pitch and every other timing in the recording.")
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
            if len(tr) >= 5 and np.max(np.abs(np.diff(tr))) > 0.06:
                j = int(np.argmax(np.abs(np.diff(tr))))
                notes.append(f"the frequency jumps by {np.abs(np.diff(tr))[j]:.2f} Hz between two analysis windows near "
                             f"{(j + 1) * 2.0:.0f} s, far more than a grid drifts: consistent with a splice, an edit or a "
                             "recorder glitch at that point")
            if len(tr) >= 5:
                notes.append(f"the frequency track runs from {tr.min():.3f} to {tr.max():.3f} Hz over "
                             f"{dur:.0f} s (a grid frequency wanders slowly like this; a perfectly "
                             "steady tone points to equipment or an inverter) and can be compared with "
                             "published grid-frequency logs")
        if dur < 60:
            notes.append(f"only {dur:.0f} s analysed; ENF log matching works best with minutes")

        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.5, meta=grid_hint(best, "mains hum"),
            constrains=["region", "date"], effort=2,
            follow_ups=follow_ups("audio", "enf"), summary="; ".join(notes) + ".")
