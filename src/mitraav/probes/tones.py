"""Tones and chimes: one probe for every tonal signal that is not a voice or an instrument.

Combines two analyses on the same audio: stable tones (telephone and alert tones from a table, or
any other stable pitch measured and named) and struck bells (tuned-bell partials, nominal pitch).
Each part reports its own findings; this probe merges them into one row.
"""
from __future__ import annotations

from pathlib import Path

from ..core.finding import Finding, FollowUp, MediaType
from ..core.probe import Probe
from .bells import ChurchBellsProbe
from .signal_tones import SignalTonesProbe


class TonesChimesProbe(Probe):
    name = "tones_chimes"
    group = "Sounds"
    clutter_sensitive = True
    masked_by = ("sirens",)
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 1200.0

    def run(self, path: Path) -> Finding:
        parts = [ChurchBellsProbe().run(path), SignalTonesProbe().run(path)]
        ran = [f for f in parts if f.status == "ok"]
        if not ran:
            first = parts[0]
            return Finding(probe=self.name, detected=False, status=first.status, detail=first.detail)
        hits = [f for f in ran if f.detected]
        if not hits:
            return Finding(probe=self.name, detected=False,
                           summary="No stable tone, alert-tone match, or struck-bell pattern found. "
                                   "Codecs, music, distance and noise hide many tones.")
        ups: list[FollowUp] = []
        for f in hits:
            ups += [u for u in f.follow_ups if all(u.name != o.name for o in ups)]
        ctx: dict = {}
        for f in hits:
            ctx.update(f.meta.get("context", {}))
        return Finding(
            probe=self.name, detected=True, meta={"context": ctx} if ctx else {},
            strength=max(f.strength for f in hits), narrows=max(f.narrows for f in hits),
            constrains=sorted({c for f in hits for c in f.constrains}),
            effort=min(f.effort for f in hits),
            regions=[r for f in hits for r in f.regions], follow_ups=ups,
            summary=" • ".join(f.summary for f in hits))
