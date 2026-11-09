"""Hidden data outside the pixels and samples: appended files, oversized or odd metadata, bad checksums,
non-empty free space, extra streams and attachments.

This is the cheapest and most reliable steganography check (it needs no statistics): many "hide a file in
a picture" tools simply append the payload, add a private chunk or write into free space. It cannot see
data woven into the picture or sound itself (see the other hidden-data probes), and a camera, editor or
phone can leave innocent padding, so every finding says how unusual it is, not what it means.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.stego_container import Observation, inspect_file


def stream_observations(path: Path) -> list[Observation]:
    """Extra tracks in a media container (ffprobe): attachments, data tracks, a second video or audio."""
    exe = shutil.which("ffprobe")
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=60)
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except (subprocess.TimeoutExpired, ValueError):
        return []
    obs: list[Observation] = []
    kinds: dict[str, int] = {}
    for s in streams:
        kind = s.get("codec_type", "?")
        disp = s.get("disposition", {})
        if kind == "video" and disp.get("attached_pic"):
            kinds["cover"] = kinds.get("cover", 0) + 1
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
        tags = s.get("tags", {})
        if kind == "attachment":
            obs.append(Observation("attachment", 0.5, f"an attached file ('{tags.get('filename', 'unnamed')}', "
                                   f"{tags.get('mimetype', 'unknown type')})", 0, 0))
        elif kind == "data":
            obs.append(Observation("data_stream", 0.35, f"a data track ({s.get('codec_tag_string', 'unknown')}), "
                                   "a container for arbitrary bytes (common for camera telemetry, GPS or timecode)", 0, 0))
    if kinds.get("video", 0) > 1:
        obs.append(Observation("extra_stream", 0.3, f"{kinds['video']} video tracks (one is normal)", 0, 0))
    if kinds.get("audio", 0) > 2:
        obs.append(Observation("extra_stream", 0.3, f"{kinds['audio']} audio tracks (an extra track can carry hidden sound)", 0, 0))
    if kinds.get("subtitle", 0):
        obs.append(Observation("subtitle", 0.15, f"{kinds['subtitle']} subtitle track(s) (text that does not appear "
                               "in the picture)", 0, 0))
    return obs


class StegoContainerProbe(Probe):
    name = "hidden_data_container"
    group = "Hidden data"
    media = (MediaType.ANY,)
    timeout = 120.0

    def run(self, path: Path) -> Finding:
        try:
            obs = inspect_file(path)
        except OSError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        obs += stream_observations(path)
        if not obs:
            return Finding(
                probe=self.name, detected=False, info=True,
                summary="No appended data, oversized or odd metadata, bad checksums or extra tracks found in "
                        "the file structure. This does not see data woven into the picture or sound itself.")
        top = max(o.severity for o in obs)
        kinds = {o.kind for o in obs}
        ups = follow_ups("stego", "appended" if "trailing_data" in kinds or "filler_not_empty" in kinds else "metadata")
        return Finding(
            probe=self.name, detected=True, strength=min(0.95, 0.35 + 0.6 * top), narrows=0.4, effort=2,
            constrains=["hidden content"], follow_ups=ups,
            summary=" • ".join(f"**{o.kind.replace('_', ' ')}**: {o.text}" for o in
                                    sorted(obs, key=lambda o: -o.severity)[:6])
            + " • These are consistent with a hiding tool, but cameras, editors and phones also leave padding "
              "and private blocks: look at what the extra bytes are before reading anything into them.")
