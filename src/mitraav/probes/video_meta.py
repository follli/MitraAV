from __future__ import annotations

import re
from pathlib import Path

from ..core.finding import Finding, FollowUp, MediaType
from ..core.frames import ffprobe
from ..core.geo import hint, macros_for_latlon
from ..core.probe import Probe
from ..core.references import follow_ups

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


class VideoMetaProbe(Probe):
    name = "video_metadata"
    group = "File and metadata"
    media = (MediaType.VIDEO,)
    timeout = 40.0

    def run(self, path: Path) -> Finding:
        info = ffprobe(path)
        if not info:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="ffprobe missing or file unreadable (sudo apt install ffmpeg)")
        tags: dict[str, str] = {}
        for src in [info.get("format", {}), *info.get("streams", [])]:
            for k, v in (src.get("tags") or {}).items():
                tags.setdefault(k.lower(), str(v))

        loc = next((v for k, v in tags.items() if "iso6709" in k or k.startswith("location")), None)
        m = _ISO6709.match(loc.strip()) if loc else None
        latlon = (float(m[1]), float(m[2])) if m else None
        taken = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
        device = " ".join(tags[k] for k in ("com.apple.quicktime.make", "com.apple.quicktime.model",
                                            "make", "model") if k in tags)
        encoder = tags.get("encoder")

        if not (latlon or taken or device):
            return Finding(probe=self.name, detected=False,
                           summary="No location, date or device tags in the container. "
                                   "Platforms usually strip these, so absence says little.")
        scores, cons, notes, fups, effort = [], [], [], [], 2
        if latlon:
            lat, lon = latlon
            scores.append((0.9, 1.0))
            cons.append("location")
            effort = 1
            notes.append(f"**location tag** (**{lat:.5f}, {lon:.5f}**)")
            fups += follow_ups("metadata", "gps")
            fups.append(FollowUp("OpenStreetMap at these coordinates",
                                 f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"))
        if taken:
            scores.append((0.5, 0.4))
            cons.append("date")
            notes.append(f"container date: **{taken}** (may be export time, not capture time)")
            fups += follow_ups("metadata", "datetime")
        if device:
            scores.append((0.5, 0.2))
            cons.append("device")
            notes.append(f"device: **{device}**")
            fups += follow_ups("metadata", "camera")
        if encoder:
            notes.append(f"encoder: {encoder}")
        s, n = max(scores, key=lambda x: x[0] * x[1])
        return Finding(probe=self.name, detected=True, strength=s, narrows=n, constrains=cons,
                       effort=effort, follow_ups=fups,
                       meta={"geo": hint(f"location tag ({latlon[0]:.2f}, {latlon[1]:.2f})",
                                         macros_for_latlon(*latlon), 0.9)} if latlon else {},
                       summary="; ".join(notes) + ". Container tags are easy to alter or strip.")
