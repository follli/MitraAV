"""Reverberation: how long sharp sounds ring on, as a hint about the space they were made in.

Finds sharp broadband onsets (a clap, a door, a knock), measures the decay after each, and reports
a reverberation time (RT60, seconds for the level to fall 60 dB) only when several decays agree.
Dry decay fits open air or a small damped room; long decay fits large enclosed spaces, stairwells,
tunnels or street canyons. Sources that fade by themselves (birds, engines) can fake it, so this is
a weak hint about the setting, not a location.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import spectrogram

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

FRAME = 0.01


def envelope_db(x: np.ndarray, sr: int, lo: float = 800.0, hi: float = 6000.0):
    n = int(sr * FRAME)
    f, t, s = spectrogram(x, fs=sr, nperseg=n, noverlap=0, mode="psd")
    hi = min(hi, 0.45 * sr)
    m = (f >= lo) & (f <= hi)
    return t, 10 * np.log10(s[m].sum(axis=0) + 1e-20)


def decay_times(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """(time of onset, RT60 estimate) for each usable sharp onset."""
    t, db = envelope_db(x, sr)
    floor = float(np.percentile(db, 10))
    out = []
    i = 5
    while i < len(db) - 120:
        rise = db[i] - np.median(db[i - 5:i - 1])
        if rise >= 25 and db[i] >= db[i - 2:i + 3].max() and db[i] - floor >= 40:
            seg = db[i:i + 150]                               # 1.5 s after the onset
            # fit the decay from 5 dB to 25 dB below the peak (ignores the noise floor)
            lo, hi = db[i] - 25, db[i] - 5
            idx = np.flatnonzero((seg <= hi) & (seg >= lo))
            if len(idx) >= 4 and idx[-1] - idx[0] < 140 and lo > floor + 5:
                k = np.arange(idx[0], idx[-1] + 1)
                slope, _ = np.polyfit(k * FRAME, seg[k], 1)
                if slope < -3.0:
                    out.append((float(t[i]), float(-60.0 / slope)))
            i += 100                                         # skip the ringing before the next
        else:
            i += 1
    return out


class ReverbProbe(Probe):
    name = "reverb_estimate"
    group = "Setting and power"
    clutter_sensitive = True
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 200.0

    def run(self, path: Path) -> Finding:
        found: list[tuple[float, float]] = []
        n = 0
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 5 or sr < 8000:
                    continue
                n += 1
                found += [(off + t, r) for t, r in decay_times(x, sr)]
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 5 s at 8 kHz")
        uniq: list[tuple[float, float]] = []
        for t, r in sorted(found):                            # overlapping chunks repeat onsets
            if not uniq or t - uniq[-1][0] > 0.3:
                uniq.append((t, r))
        rts = np.array([r for _, r in uniq])
        if len(rts) < 3:
            return Finding(probe=self.name, detected=False,
                           summary=f"Only {len(rts)} sharp onset(s) with a measurable decay; need at "
                                   "least 3 to estimate reverberation.")
        q1, med, q3 = (float(v) for v in np.percentile(rts, [25, 50, 75]))
        if q3 / max(q1, 0.05) > 2.2:
            return Finding(probe=self.name, detected=False,
                           summary=f"{len(rts)} decays measured but they disagree (RT60 from "
                                   f"{rts.min():.2f} to {rts.max():.2f} s), so no reliable estimate.")
        if med < 0.3:
            kind = "short: consistent with open air or a small, damped space"
        elif med < 0.9:
            kind = "moderate: consistent with an ordinary room or a space with some reflecting walls"
        else:
            kind = ("long: consistent with a large enclosed space, stairwell, tunnel or a street "
                    "canyon between hard buildings")
        regions = [dict(label=f"decay at {t:.0f} s (RT60 about {r:.2f} s)", t0=max(0.0, t - 0.5),
                        t1=t + 2.0, f0=500.0, f1=8000.0) for t, r in uniq[:6]]
        return Finding(
            probe=self.name, detected=True, strength=min(0.8, 0.25 + 0.08 * len(rts)), narrows=0.1,
            constrains=["setting"], effort=1, regions=regions,
            follow_ups=follow_ups("audio", "reverb"),
            summary=f"{len(rts)} sharp sounds decay consistently: reverberation time about "
                    f"**{med:.2f} s** (middle half {q1:.2f} to {q3:.2f} s), {kind}. Sources that fade "
                    "on their own (birds, engines) can imitate this; treat as a hint about the "
                    "setting only.")
