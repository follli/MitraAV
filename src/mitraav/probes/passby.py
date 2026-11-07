from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, percentile_filter
from scipy.optimize import curve_fit
from scipy.signal import spectrogram
from scipy.stats import spearmanr

from ..core.audio import AudioError, load_audio, sound_start
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

FRAME = 0.5          # s per analysis frame
FLO, FHI = 150.0, 3500.0
MIN_EXCESS_DB = 6.0  # loudness bump above the slow baseline
MAX_EXCESS_DB = 45.0
MAX_TONAL = 0.5          # median share of power in the 3 strongest bins above which it is a tone
MIN_DUR, MAX_DUR = 5.0, 150.0


def band_series(x: np.ndarray, sr: int):
    """Per-frame band power (linear), spectral centroid (Hz) and frame times."""
    n = int(sr * FRAME)
    f, t, s = spectrogram(x, fs=sr, nperseg=n, noverlap=0, window="hann", mode="psd")
    hi = min(FHI, 0.45 * sr)
    m = (f >= FLO) & (f <= hi)
    s = s[m]
    p = s.sum(axis=0) + 1e-20
    cen = (f[m][:, None] * s).sum(axis=0) / p
    top3 = np.sort(s, axis=0)[-3:].sum(axis=0)
    band_series.tonal = top3 / p              # share of the power in the 3 strongest bins
    return p, cen, t


def _model(t, a0, t0, tau):
    return a0 / np.sqrt(1 + ((t - t0) / tau) ** 2)


def analyse(x: np.ndarray, sr: int) -> dict | None:
    p, cen, t = band_series(x, sr)
    tonal = band_series.tonal
    if len(t) < 30:
        return None
    env = median_filter(10 * np.log10(p), size=5, mode="nearest")
    base = percentile_filter(env, 20, size=min(len(env), int(120 / FRAME)), mode="reflect")
    ex = env - base
    edge = int(3 / FRAME)                      # fades at the start and end are not pass-bys
    ex[:edge] = 0
    ex[-edge:] = 0
    i = int(np.argmax(ex))
    if ex[i] < MIN_EXCESS_DB or ex[i] > MAX_EXCESS_DB:     # implausibly large: a level jump, not a pass
        return None
    # extent: contiguous frames around the peak still above 3 dB
    lo = i
    while lo > 0 and ex[lo - 1] > 3.0:
        lo -= 1
    hi = i
    while hi < len(ex) - 1 and ex[hi + 1] > 3.0:
        hi += 1
    dur = float(t[hi] - t[lo])
    if not MIN_DUR <= dur <= MAX_DUR or i - lo < 4 or hi - i < 4:
        return None
    # a siren, horn or engine drone is one strong tone, not broadband noise like a jet
    if float(np.median(tonal[lo:hi + 1])) > MAX_TONAL:
        return None
    # rises before the peak, falls after it
    if spearmanr(t[lo:i + 1], ex[lo:i + 1])[0] < 0.6 or spearmanr(t[i:hi + 1], ex[i:hi + 1])[0] > -0.6:
        return None
    # centroid higher while approaching than while receding
    k = max(3, (hi - lo) // 4)
    c_before = float(np.median(cen[max(lo, i - 2 * k):max(lo + 1, i - k // 2)]))
    c_after = float(np.median(cen[min(hi, i + k // 2):min(hi + 1, i + 2 * k)]))
    ratio = c_before / max(c_after, 1.0)
    # distance/speed from the envelope shape (amplitude ~ 1/slant range)
    w = slice(max(0, lo - 4), min(len(t), hi + 5))
    amp = np.sqrt(np.maximum(p[w] - 10 ** (base[w] / 10), 1e-20))
    tau = r2 = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_model, t[w], amp, p0=[amp.max(), t[i], dur / 4],
                                bounds=([0, t[lo], 0.5], [np.inf, t[hi], 120.0]), maxfev=4000)
        res = float(np.sum((amp - _model(t[w], *popt)) ** 2))
        r2 = 1 - res / max(float(np.sum((amp - amp.mean()) ** 2)), 1e-20)
        if r2 >= 0.7:
            tau = float(popt[2])
    except (RuntimeError, ValueError):
        pass
    return dict(t_lo=float(t[lo]), t_hi=float(t[hi]), t_peak=float(t[i]), excess=float(ex[i]), dur=dur, ratio=float(ratio),
                c_before=c_before, c_after=c_after, tau=tau, r2=r2)


class BroadbandPassbyProbe(Probe):
    name = "broadband_passby"
    group = "Aircraft and motion"
    clutter_sensitive = True
    masked_by = ("sirens",)
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 40.0

    def run(self, path: Path) -> Finding:
        try:
            off = sound_start(path)       # times below are reported in file seconds
            x, sr = load_audio(path, off)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if len(x) / sr < 20 or sr < 4000:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Need at least 20 s of audio sampled at 4 kHz or more")
        r = analyse(x, sr)
        if r:
            r = {**r, **{k: r[k] + off for k in ("t_lo", "t_hi", "t_peak")}}
        if not r:
            return Finding(
                probe=self.name, detected=False,
                summary="No smooth loudness rise and fall (a pass-by) above the slow background. "
                        "Wind, traffic and speech can hide a distant aircraft.")

        doppler = r["ratio"] >= 1.04
        strength = min(1.0, 0.2 + 0.4 * min(1.0, (r["excess"] - MIN_EXCESS_DB) / 15))
        if doppler:
            strength = min(1.0, strength + 0.3 * min(1.0, (r["ratio"] - 1.04) / 0.2))
        notes = [f"loudness bump of about **{r['excess']:.0f} dB** peaking near **{r['t_peak']:.0f} s**, "
                 f"lasting about {r['dur']:.0f} s"]
        if doppler:
            notes.append(f"spectral centre of mass drops from about {r['c_before']:.0f} to "
                         f"{r['c_after']:.0f} Hz across the peak, consistent with a **moving "
                         "broadband source** such as a jet")
        else:
            notes.append("no clear spectral shift across the peak, so any passing source "
                         "(car, train, wind gust) fits equally well")
        cons = ["time"]
        if r["tau"]:
            notes.append(f"envelope shape is consistent with distance/speed of about "
                         f"{r['tau']:.0f} s (fit R2 {r['r2']:.2f}, treat as a range); multiply by a speed from "
                         "ADS-B or a Doppler tone to get closest slant distance")
            cons.append("distance")
        notes.append("the loudness peak is when the sound arrives, so shift ADS-B closest-approach "
                     "times later by slant distance / 343 m/s before comparing")
        region = dict(label=f"loudness peak near {r['t_peak']:.0f} s",
                      t0=max(0.0, r["t_lo"] - 15), t1=r["t_hi"] + 15, f0=FLO, f1=min(FHI, 0.45 * sr))
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.25,
                       regions=[region],
                       constrains=cons, effort=3, follow_ups=follow_ups("audio", "doppler"),
                       summary="; ".join(notes) + ".")
