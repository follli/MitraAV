from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..core.finding import Finding, FollowUp, MediaType
from ..core.geo import hint, macros_for_latlon
from ..core.probe import Probe
from ..core.references import follow_ups

GPS_IFD, EXIF_IFD = 0x8825, 0x8769


def _to_deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    v = d + m / 60 + s / 3600
    return -v if ref in ("S", "W") else v


class MetadataProbe(Probe):
    name = "metadata"
    group = "File and metadata"
    media = (MediaType.IMAGE,)
    timeout = 5.0
    frames_ok = False  # extracted frames carry no EXIF
    def run(self, path: Path) -> Finding:
        with Image.open(path) as img:
            exif = img.getexif()
            gps = dict(exif.get_ifd(GPS_IFD))
            sub = dict(exif.get_ifd(EXIF_IFD))
            base = dict(exif)

        make, model, software = base.get(0x010F), base.get(0x0110), base.get(0x0131)
        taken = sub.get(36867) or base.get(0x0132)

        latlon = None
        if gps.get(2) and gps.get(4):
            try:
                latlon = (_to_deg(gps[2], gps.get(1, "N")), _to_deg(gps[4], gps.get(3, "E")))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        if not (latlon or taken or make or model):
            return Finding(
                probe=self.name, detected=False,
                summary="No usable EXIF. Common after uploads to platforms that strip "
                        "metadata, so absence says little.")

        scores, constrains, notes, fups, effort = [], [], [], [], 2
        if latlon:
            lat, lon = latlon
            scores.append((0.9, 1.0))
            constrains.append("location")
            notes.append(f"**GPS tags** present (**{lat:.5f}, {lon:.5f}**)")
            fups += follow_ups("metadata", "gps")
            fups.append(FollowUp(
                "OpenStreetMap at these coordinates",
                f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"))
            effort = 1
        if taken:
            scores.append((0.6, 0.4))
            constrains.append("date")
            notes.append(f"capture time tag: **{taken}**")
            fups += follow_ups("metadata", "datetime")
        if make or model:
            scores.append((0.5, 0.2))
            constrains.append("device")
            notes.append(f"device: **{' '.join(str(x) for x in (make, model) if x)}**")
            fups += follow_ups("metadata", "camera")
        if software:
            notes.append(f"software tag: {software} (metadata may have been rewritten)")

        strength, narrows = max(scores, key=lambda s: s[0] * s[1])
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=narrows,
            constrains=constrains, effort=effort, follow_ups=fups,
            meta={"geo": hint(f"GPS tag ({latlon[0]:.2f}, {latlon[1]:.2f})", macros_for_latlon(*latlon), 0.9)}
            if latlon else {},
            summary="; ".join(notes) + ". Consistent with the file's origin unless "
                    "edited or spoofed; EXIF is easy to alter.")
