from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.tracks import Track, trace

C = 343.0                    # speed of sound, m/s
WINDOWS = (0.12, 0.35, 0.9)  # analysis window lengths (s): fast sweeps need short, slow curves long
FMAX = 1500.0                # aircraft/engine tones live here; above it birdsong dominates
MIN_NARROW_DB = 8.5          # ridge must stand this far above neighbouring bins
MAX_SLOPE = 0.09             # 90th percentile |d ln f / dt| (per s)
MAX_JITTER = 0.013           # 90th percentile deviation from a running median
MIN_SLANT_M = 100.0          # a 'closest distance' below this is a near-step fit, not a measurement
MAX_UP = 0.15                # lone tracks that step up and down are machinery, not glides
MIN_RATIO = 1.08             # smaller glides are indistinguishable from drifting noise


def doppler_model(t, f0, beta, t0, tau):
    """Source passing at constant speed beta*C; tau = closest-approach distance / speed."""
    u = (t - t0) / tau
    return f0 / (1 + beta * u / np.sqrt(1 + u * u))


def fit_doppler(t: np.ndarray, f: np.ndarray):
    if len(t) > 300:
        i = np.linspace(0, len(t) - 1, 300).astype(int)
        t, f = t[i], f[i]
    k = max(3, len(f) // 10)
    hi, lo = float(np.median(f[:k])), float(np.median(f[-k:]))
    dur = float(t[-1] - t[0])
    g = np.gradient(median_filter(f, size=5, mode="nearest"), t)
    p0 = [np.sqrt(hi * lo), float(np.clip((hi - lo) / (hi + lo), 0.02, 0.7)),
          float(t[int(np.argmin(g))]), max(dur / 6, 0.2)]
    bounds = ([f.min() * 0.5, 0.005, t[0] - dur, 0.05], [f.max() * 2, 0.9, t[-1] + dur, 120.0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(doppler_model, t, f, p0=p0, bounds=bounds,
                                x_scale="jac", maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    ss_res = float(np.sum((f - doppler_model(t, *popt)) ** 2))
    ss_tot = float(np.sum((f - f.mean()) ** 2))
    return popt, 1 - ss_res / max(ss_tot, 1e-12)


def level_peaks_at_t0(tr: Track, t0: float, dur: float) -> bool:
    """A real pass-by is loudest near closest approach and quieter at both ends."""
    lv = median_filter(tr.level, size=5, mode="nearest")
    k = max(3, len(lv) // 10)
    i = int(np.argmax(lv))
    ends = max(float(np.median(lv[:k])), float(np.median(lv[-k:])))
    return abs(float(tr.t[i]) - t0) <= 0.35 * dur and float(lv[i]) - ends >= 3.0


def clean_ridge(tr: Track) -> bool:
    """A real tone ridge is narrow and smooth. Chains hopping through noise bursts, transients
    or wind are not: they jitter, change pitch implausibly fast, or barely exceed their
    neighbouring bins. (Thresholds from the real demo clip and the synthetic samples.)"""
    if tr.narrow < MIN_NARROW_DB:
        return False
    f, t = tr.f, tr.t
    dt = np.diff(t)
    ok = dt <= 3 * float(np.median(dt)) + 1e-9
    if not ok.any():
        return False
    slope = np.abs(np.diff(np.log(f)))[ok] / np.maximum(dt[ok], 1e-9)
    jit = np.abs(np.log(f) - np.log(median_filter(f, size=9, mode="nearest")))
    return float(np.percentile(slope, 90)) <= MAX_SLOPE and float(np.percentile(jit, 90)) <= MAX_JITTER


def _up_fraction(f: np.ndarray) -> float:
    """Share of total pitch movement that goes upward: ~0 for a glide, higher for lines that
    step up and down (machinery, speed-controlled fans)."""
    d = np.diff(np.log(median_filter(f, size=5, mode="nearest")))
    return float(d[d > 0].sum() / max(np.abs(d).sum(), 1e-9))


def evaluate(tr: Track):
    if not clean_ridge(tr):
        return None
    f = median_filter(tr.f, size=5, mode="nearest")
    t = tr.t
    k = max(3, len(f) // 10)
    hi, lo = float(np.median(f[:k])), float(np.median(f[-k:]))
    if lo <= 0 or not MIN_RATIO <= hi / lo <= 3.0:
        return None
    if not float(spearmanr(t, f)[0]) <= -0.85:
        return None
    fit = fit_doppler(t, f)
    if not fit:
        return None
    popt, r2 = fit
    if r2 < 0.85:
        return None
    dur = float(t[-1] - t[0])
    if dur < 4.0:
        return None
    t0, tau = float(popt[2]), float(popt[3])
    return dict(hi=hi, lo=lo, r2=r2, beta=float(popt[1]), beta_lb=(hi - lo) / (hi + lo),
                t0=t0, tau=tau, dur=dur, t_start=float(t[0]), t_end=float(t[-1]), support=1, narrow=float(tr.narrow), up=_up_fraction(f),
                sat=bool(tau <= 0.5 * dur and t[0] < t0 < t[-1] and popt[1] * C * tau >= MIN_SLANT_M),
                level_ok=level_peaks_at_t0(tr, t0, dur))


def _same_event(e: dict, k: dict) -> bool:
    """Same pass-by: closest-approach times agree, or the spans overlap with a matching
    pitch ratio (harmonics of one source glide by the same ratio)."""
    if abs(e["t0"] - k["t0"]) <= max(6.0, 0.25 * max(e["dur"], k["dur"])):
        return True
    ov = min(e["t_end"], k["t_end"]) - max(e["t_start"], k["t_start"])
    if ov < 0.5 * min(e["dur"], k["dur"]):
        return False
    return abs(np.log(e["hi"] / e["lo"]) - np.log(k["hi"] / k["lo"])) < 0.1


def dedupe(evals: list[dict]) -> list[dict]:
    """Overlapping analysis chunks see the same track twice; that is not harmonic support."""
    out: list[dict] = []
    for e in sorted(evals, key=lambda e: -e["r2"]):
        if not any(abs(e["t_start"] - k["t_start"]) < 3.0 and abs(e["t_end"] - k["t_end"]) < 3.0
                   and abs(np.log(e["hi"] / k["hi"])) < 0.04 for k in out):
            out.append(e)
    return out


def group_events(evals: list[dict]) -> list[dict]:
    """Harmonics and repeated detections of one event are merged; support counts them."""
    kept: list[dict] = []
    for e in sorted(evals, key=lambda e: -(e["r2"] * min(e["dur"], 10))):
        for k in kept:
            if _same_event(e, k):
                k["support"] += 1
                break
        else:
            kept.append(e)
    # a lone track (no harmonic partner) must be a long, clear glide to count
    kept = [k for k in kept if k["support"] > 1 or (k["dur"] >= 8.0 and k["hi"] / k["lo"] >= 1.15)]
    return sorted(kept, key=lambda e: e["t0"])


def describe(i: int, e: dict) -> str:
    s = f"**event {i}**: closest approach heard near {e['t0']:.0f} s, pitch {e['hi']:.0f} to {e['lo']:.0f} Hz"
    if e["support"] > 1:
        s += f" ({e['support']} matching tracks, e.g. harmonics)"
    if e["sat"]:
        v = e["beta"] * C
        s += (f", about **{v:.0f} m/s** ({v * 3.6:.0f} km/h), closest slant distance "
              f"about **{v * e['tau']:.0f} m**")
    else:
        v = e["beta_lb"] * C
        s += (f", speed at least **{v:.0f} m/s** ({v * 3.6:.0f} km/h); the curve is too flat to "
              "pin down speed or distance")
    return s


class DopplerProbe(Probe):
    name = "doppler_pass"
    group = "Aircraft and motion"
    clutter_sensitive = True
    masked_by = ("sirens",)
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 900.0

    def run(self, path: Path) -> Finding:
        tracks: list[Track] = []
        n_chunks = 0
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 5 or sr < 2000:
                    continue
                n_chunks += 1
                for w in WINDOWS:
                    for tr in trace(x, sr, nperseg=int(2 ** round(np.log2(sr * w))), fmax=FMAX):
                        tr.t = tr.t + off          # report file-absolute time
                        tracks.append(tr)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n_chunks:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need at least 5 s of non-silent audio "
                                  "sampled at 2 kHz or more")
        if not tracks:
            return Finding(
                probe=self.name, detected=False,
                summary="No sustained moving tone found. Jet pass-bys are broadband and often "
                        "show no clean tone; propellers, rotors, horns and engines do. Very faint "
                        "curves can also be hidden: boost levels and check the spectrogram by eye.")
        evals = [e for e in map(evaluate, tracks) if e]
        if not evals:
            return Finding(
                probe=self.name, detected=False,
                summary=f"{len(tracks)} moving tone track(s) found, none shaped like a pass-by "
                        "(a smooth downward glide of 8-200% pitch change).")

        events = group_events(dedupe(evals))
        if not events:
            return Finding(
                probe=self.name, detected=False,
                summary=f"{len(evals)} glide(s) found but none long or consistent enough to be a "
                        "pass-by (isolated tracks need 8 s and a 15% pitch change).")
        strong = [e for e in events if e["support"] > 1 or (e["sat"] and e["up"] <= MAX_UP)]
        weak = [e for e in events if e not in strong]
        if not strong:       # keep the best lone track so something is reported, flagged weak
            strong, weak = [max(events, key=lambda e: e["r2"] * min(e["dur"], 10))], []
            weak = [e for e in events if e not in strong]
        events = strong
        best = max(events, key=lambda e: e["r2"] * min(e["dur"], 10))
        quality = min(1.0, max(0.0, (best["r2"] - 0.85) / 0.15))
        strength = 0.25 + 0.5 * quality + 0.25 * min(1.0, best["dur"] / 8)
        if best["level_ok"]:
            strength = min(1.0, strength + 0.1)
        if not best["sat"]:
            strength *= 0.6
        if len(events) > 1:
            strength = min(1.0, strength + 0.05 * (len(events) - 1))

        notes = [describe(i + 1, e) for i, e in enumerate(events)]
        if weak:
            notes.append("weaker candidates (single flat tracks, check the spectrogram by eye; "
                         "stepping machinery lines can mimic them) near "
                         + ", ".join(f"{e['t0']:.0f} s" for e in weak))
        cons = ["source type"] + (["speed", "distance"] if any(e["sat"] for e in events) else [])
        if len(events) > 1:
            gaps = [f"{b['t0'] - a['t0']:.0f} s" for a, b in zip(events, events[1:])]
            notes.append("offset between strong events' closest-approach times: " + ", ".join(gaps))
            cons.append("time")
        if not best["sat"]:
            notes.append("a siren sweep, slowing engine or speed-changed audio fits a flat "
                         "glide equally well")
        notes.append("Audio alone cannot identify the aircraft. Match speed, slant distance "
                     "(not ground distance) and the offset against ADS-B tracks, shifting each "
                     "track's closest-approach time later by slant distance / 343 m/s")
        notes.append("fit ignores sound-travel delay in the curve shape, so treat numbers as "
                     "ranges")
        regions = [dict(label=f"event {i + 1}: {e['hi']:.0f}→{e['lo']:.0f} Hz near {e['t0']:.0f} s",
                        t0=max(0.0, e["t_start"] - 15), t1=e["t_end"] + 15,
                        f0=max(20.0, 0.7 * e["lo"]), f1=1.4 * e["hi"])
                   for i, e in enumerate(events)]
        regions += [dict(label=f"weak candidate near {e['t0']:.0f} s",
                         t0=max(0.0, e["t_start"] - 15), t1=e["t_end"] + 15,
                         f0=max(20.0, 0.7 * e["lo"]), f1=1.4 * e["hi"]) for e in weak]
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.3,
                       regions=regions,
                       constrains=cons, effort=3, follow_ups=follow_ups("audio", "doppler"),
                       summary=" • ".join(notes) + ".")
