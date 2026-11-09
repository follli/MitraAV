"""Time-of-day and season cues in the sound: clock strikes, dawn-chorus birdsong, crowing roosters, insects
and frogs.

These are weak, indirect clues (a rooster can crow at noon, birds sing at dusk too), so they are phrased
as what the sound is consistent with. A clock striking is the strongest: n strikes in a row fit "n
o'clock" on a 12-hour striking clock. It reads what the sound overview and the tones-and-chimes probe
already found, so it adds no new analysis of the audio.
"""
from __future__ import annotations

from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.probe import Probe


def strike_groups(strikes: list[float], max_gap: float = 8.0) -> list[list[float]]:
    out: list[list[float]] = []
    for t in sorted(strikes):
        if out and t - out[-1][-1] <= max_gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def hour_cues(bells: dict) -> list[str]:
    """Clock-striking readings from a series of single-pitch strikes (change ringing is skipped)."""
    if not bells or bells.get("distinct", 1) > 2:
        return []
    notes = []
    for g in strike_groups(bells.get("strikes", [])):
        gaps = [b - a for a, b in zip(g, g[1:])]
        if 2 <= len(g) <= 12 and gaps and max(gaps) / max(min(gaps), 0.1) < 1.6:
            notes.append(f"{len(g)} evenly spaced bell strikes near {g[0]:.0f} s are consistent with a clock striking "
                         f"**{len(g)} o'clock** on a 12-hour striking clock (or the {len(g)}th hour after noon or midnight)")
        elif len(g) == 1:
            continue
    return notes[:2]


class TimeCuesProbe(Probe):
    name = "time_cues"
    group = "Sounds"
    order = 70                       # after the sound overview and the bells probe
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 10.0

    def run(self, path: Path) -> Finding:
        sound = self.context.get("sound") or {}
        labels = sound.get("labels") or {}
        shares = sound.get("shares") or {}
        notes = hour_cues(self.context.get("bells") or {})
        strength = 0.5 if notes else 0.0
        low = {k: v for k, v in labels.items()}

        def has(*words):
            return sum(v for k, v in low.items() if any(w in k for w in words))

        if has("rooster", "cock-a-doodle", "crowing"):
            notes.append("a crowing rooster is heard: roosters crow most around dawn but also through the day")
            strength = max(strength, 0.25)
        if shares.get("birds", 0) >= 0.3:
            notes.append(f"birdsong fills about {shares['birds'] * 100:.0f}% of the recording: consistent with the dawn or dusk "
                         "chorus, or with spring and early summer (in winter and at midday birds are much quieter)")
            strength = max(strength, 0.3)
        if has("cicada"):
            notes.append("cicadas are heard: consistent with a warm season, usually in daylight or at dusk")
            strength = max(strength, 0.3)
        elif has("cricket", "insect"):
            notes.append("crickets or other insects are heard: consistent with a warm evening or night")
            strength = max(strength, 0.25)
        if has("frog"):
            notes.append("frogs are heard: consistent with a mild, damp evening or night")
            strength = max(strength, 0.25)
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No clock strikes, dawn-chorus birdsong, roosters, insects or frogs that would hint at the "
                                   "time of day or season (this reads the sound overview, so it needs that add-on).")
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.15, effort=1,
                       constrains=["date", "time"], summary=" • ".join(notes)
                       + " • Indirect clues only; compare with shadows, light and the file's own dates.")
