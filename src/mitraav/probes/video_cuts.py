"""Scene cuts: is the video one continuous shot or an edited montage?

Counts hard cuts with ffmpeg's scene detector on a small copy of the first ten minutes. Many cuts, or a
cut every few seconds, is consistent with an edited compilation (news package, social-media mashup),
which matters for provenance: the clips may come from different places and days. A single long shot
points to raw footage. Fades, whip pans and fast action can add or hide cuts.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.probe import Probe

SCENE = 0.35
SHOWINFO = re.compile(r"pts_time:([\d.]+)")


def cut_times(path: Path, threshold: float = SCENE, limit: float = 600.0) -> list[float] | None:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-v", "info", "-t", str(limit), "-i", str(path), "-an", "-vf",
                            f"scale=160:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    return [float(m) for m in SHOWINFO.findall(r.stderr)]


class VideoCutsProbe(Probe):
    name = "video_cuts"
    group = "File and metadata"
    media = (MediaType.VIDEO,)
    timeout = 360.0

    def run(self, path: Path) -> Finding:
        cuts = cut_times(path)
        if cuts is None:
            return Finding(probe=self.name, detected=False, status="not_run", detail="ffmpeg is needed to find scene cuts")
        if len(cuts) < 2:
            return Finding(probe=self.name, detected=False, info=True,
                           summary=f"{len(cuts)} hard cut(s) found in the first ten minutes: consistent with one continuous shot.")
        span = max(cuts[-1] - cuts[0], 1.0)
        gaps = [b - a for a, b in zip(cuts, cuts[1:])]
        rate = len(cuts) / max(cuts[-1], 1.0)
        kind = "an edited montage or compilation" if rate >= 0.08 or len(cuts) >= 6 else "an edit with a few joins"
        regions = [dict(label=f"cut near {t:.0f} s", t0=max(0.0, t - 2), t1=t + 3, f0=0.0, f1=8000.0, t_play=max(0.0, t - 2))
                   for t in cuts[:12]]
        return Finding(probe=self.name, detected=True, strength=min(0.8, 0.35 + 0.05 * len(cuts)), narrows=0.25, effort=1,
                       constrains=["origin"], regions=regions,
                       summary=f"**{len(cuts)} hard cuts** over {span:.0f} s (about one every {sum(gaps) / len(gaps):.0f} s): "
                               f"consistent with {kind}, so the shots may come from different places and times. "
                               "Check each shot separately (use Scan this frame).")
