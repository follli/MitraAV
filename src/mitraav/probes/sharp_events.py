"""A timeline of sharp sounds: bangs, claps, knocks, shots, fireworks, hammering.

Finds sudden loud onsets and describes their rhythm: a regular series (machinery, hammering, a drum, a
clock), a burst (several in under two seconds), or a loud sound followed by a quieter copy a fraction of a
second later (an echo, or the crack and boom of a supersonic round). The delay is converted to the extra
distance sound travelled (343 m/s), but only as arithmetic: a delay alone cannot say which it is.
Nothing is matched against any catalogue of incidents; the sound overview names what the class sounds
like.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import spectrogram

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.probe import Probe

C = 343.0
FRAME = 0.005


def find_onsets(x: np.ndarray, sr: int, rise_db: float = 20.0, above_floor_db: float = 25.0) -> list[tuple[float, float]]:
    """[(time s, level dB above the file's quiet floor)] of sudden, broadband, quickly fading sounds.

    A bang rises within a few milliseconds, spreads across the spectrum (a bird's whistle or a vowel
    does not) and dies away within about 100 ms (a spoken syllable lasts longer)."""
    n = int(sr * FRAME)
    f, t, s = spectrogram(x, fs=sr, nperseg=n, noverlap=0, mode="psd")
    band = (f >= 300) & (f <= min(6000, 0.45 * sr))
    sb = s[band]
    db = 10 * np.log10(sb.sum(axis=0) + 1e-20)
    floor = float(np.percentile(db, 10))
    flat = np.exp(np.log(sb + 1e-20).mean(axis=0)) / (sb.mean(axis=0) + 1e-20)          # spectral flatness 0..1
    out: list[tuple[float, float]] = []
    back = int(0.3 / FRAME)
    for i in range(back, len(db) - 24):
        base = float(np.median(db[i - back:i - 2]))
        if (db[i] - base >= rise_db and db[i] - floor >= above_floor_db and db[i] == db[max(0, i - 6):i + 7].max()
                and db[i] - db[i - 2] >= 12 and flat[i] >= 0.12 and db[i + 10:i + 24].min() <= db[i] - 12):
            if not out or t[i] - out[-1][0] >= 0.04:
                out.append((float(t[i]), float(db[i] - floor)))
    return out


def describe(events: list[tuple[float, float]]) -> list[str]:
    notes = []
    times = [t for t, _ in events]
    gaps = np.diff(times)
    for i in range(len(times) - 3):
        g = gaps[i:i + 3]
        if g.min() > 0.15 and g.std() / g.mean() < 0.12:
            run = 4
            while i + run < len(times) and abs(gaps[i + run - 1] - g.mean()) / g.mean() < 0.2:
                run += 1
            notes.append(f"a **regular series of {run} sharp sounds** about every {g.mean():.2f} s near {times[i]:.0f} s "
                         "(machinery, hammering, drumming, a clock or a repeated alarm)")
            break
    for i in range(len(times) - 2):
        if times[i + 2] - times[i] <= 1.5:
            notes.append(f"a **burst of {sum(1 for t in times if times[i] <= t <= times[i] + 1.5)} sharp sounds** within 1.5 s "
                         f"near {times[i]:.1f} s (fireworks, a volley, a slammed door series or rapid hammering)")
            break
    for (t0, l0), (t1, l1) in zip(events, events[1:]):
        d = t1 - t0
        if 0.08 <= d <= 3.0 and l0 - l1 >= 6:
            notes.append(f"a loud sound near {t0:.1f} s followed {d:.2f} s later by one {l0 - l1:.0f} dB quieter: if it is the "
                         f"same event, the second path is about {d * C:.0f} m longer (an echo, or the crack and boom of a "
                         "supersonic round); it cannot say which")
            break
    return notes


class SharpEventsProbe(Probe):
    name = "sharp_events"
    group = "Sounds"
    order = 60
    clutter_sensitive = True
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 300.0

    def run(self, path: Path) -> Finding:
        events: list[tuple[float, float]] = []
        try:
            for i, (off, x, sr) in enumerate(iter_chunks(path)):
                if i >= 3:
                    break
                if sr < 8000 or len(x) / sr < 3:
                    continue
                events += [(off + t, lv) for t, lv in find_onsets(x, sr)]
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        events.sort()
        uniq = [e for k, e in enumerate(events) if k == 0 or e[0] - events[k - 1][0] > 0.04]
        if len(uniq) < 3:
            return Finding(probe=self.name, detected=False, info=True,
                           summary=f"{len(uniq)} sharp loud sound(s) found: no rhythm, burst or echo pattern to report.")
        notes = describe(uniq)
        named = [k for k, v in ((self.context.get("sound") or {}).get("labels") or {}).items()
                 if any(w in k for w in ("gunshot", "gunfire", "explosion", "fireworks", "firecracker", "bang", "slam", "knock"))]
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary=f"{len(uniq)} sharp loud sounds found but with no regular rhythm, burst or echo pair.")
        if named:
            notes.append("the sound overview names: " + ", ".join(named[:3]))
        regions = [dict(label=f"sharp sound near {t:.1f} s", t0=max(0.0, t - 1), t1=t + 2, f0=0.0, f1=8000.0, t_play=max(0.0, t - 1))
                   for t, _ in uniq[:12]]
        return Finding(probe=self.name, detected=True, strength=0.4, narrows=0.1, effort=1, regions=regions,
                       summary=f"**{len(uniq)} sharp loud sounds**: " + " • ".join(notes)
                       + " • Timing only; this does not identify the source or match any incident.")
