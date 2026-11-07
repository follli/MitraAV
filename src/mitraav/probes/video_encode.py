"""Video encode fingerprint: where the file probably came from and what has been done to it.

Reads container and stream structure with ffprobe (codec, frame rate, GOP pattern, handler and
encoder tags). Platforms and tools leave recognisable marks; none of this says where the video was
filmed, and platforms strip or rewrite most of it, so every line is a weak lead for provenance.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.frames import ffprobe
from ..core.probe import Probe
from ..core.references import follow_ups

_HANDLER_DATE = re.compile(r"Created on:\s*(\d{2}/\d{2}/\d{4})")


def _rate(s: str | None) -> float | None:
    try:
        v = float(Fraction(s))
        return v if v > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def gop_pattern(path: Path, seconds: int = 20) -> tuple[int | None, bool]:
    """(distance between keyframes in frames, uses B-frames) from the first seconds."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None, False
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "frame=pict_type", "-read_intervals", f"%+{seconds}", "-of", "csv=p=0",
                            str(path)], capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return None, False
    types = [t.strip().split(",")[0] for t in r.stdout.split() if t.strip()]
    keys = [i for i, t in enumerate(types) if t == "I"]
    gop = None
    if len(keys) >= 2:
        d = [b - a for a, b in zip(keys, keys[1:])]
        gop = int(max(set(d), key=d.count))
    return gop, "B" in types


def fingerprint(info: dict, gop: int | None, bframes: bool) -> tuple[list[str], int]:
    """Plain-language observations, and a count of the specific ones."""
    fmt = info.get("format", {})
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    au = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not vs:
        return [], 0
    tags = {k.lower(): str(v) for s in [fmt, vs, au or {}] for k, v in (s.get("tags") or {}).items()}
    handlers = tags.get("handler_name", "")
    notes, specific = [], 0

    codec = vs.get("codec_name", "?")
    desc = f"**{codec} {vs.get('width')}x{vs.get('height')}**"
    fps = _rate(vs.get("avg_frame_rate"))
    if fps:
        desc += f" at {fps:.3g} fps"
    notes.append(desc)

    if "google inc" in handlers.lower():
        m = _HANDLER_DATE.search(handlers)
        notes.append("the handler tag 'ISO Media file produced by Google Inc.' is typical of "
                     "**YouTube**-processed video" + (f", stamped {m[1]} (month/day/year, when YouTube "
                                                  "processed it, not when it was filmed)" if m else ""))
        specific += 1
    if "core media" in handlers.lower() or any(k.startswith("com.apple.quicktime") for k in tags):
        notes.append("**Apple** QuickTime/Core Media tags suggest the file came from an iPhone, iPad "
                     "or a Mac app, possibly unmodified")
        specific += 1
    enc = " ".join(v for k, v in tags.items() if k == "encoder")
    if "lavf" in enc.lower() or "lavc" in enc.lower() or "libx264" in enc.lower():
        notes.append("an **FFmpeg**/libx264 encoder tag means the file was re-encoded or remuxed with "
                     "FFmpeg-based software (many platforms, converters and editors)")
        specific += 1
    if not any(k in tags for k in ("creation_time", "com.apple.quicktime.creationdate")):
        notes.append("no creation time in the container (stripped by a platform or tool)")

    r, a = _rate(vs.get("r_frame_rate")), fps
    if r and a and abs(r - a) / r > 0.02:
        notes.append("the frame rate is variable, common for phone recordings and screen captures")
        specific += 1
    elif fps:
        if abs(fps - 25) < 0.1:
            notes.append("25 fps is the PAL standard (Europe and many other regions) and also a "
                         "common platform output rate; weak regional hint")
        elif abs(fps - 29.97) < 0.01:
            notes.append("29.97 fps is the NTSC rate (North America, Japan and others) and a phone "
                         "default; weak regional hint")
    if gop:
        extra = ""
        if gop == 250:
            extra = ": 250 is the x264 default, typical of a software re-encode"
        elif gop in (30, 60):
            extra = ": a one-second GOP is typical of camera and phone hardware encoders"
        notes.append(f"keyframes about every {gop} frames{extra}" + ("; B-frames in use" if bframes else ""))
    if au:
        notes.append(f"audio {au.get('codec_name')} {au.get('sample_rate', '?')} Hz, "
                     f"{au.get('channels', '?')} ch")
    if vs.get("color_primaries") in ("bt2020", "smpte2085") or vs.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        notes.append("HDR colour tags (a recent phone or camera, or an HDR export)")
        specific += 1
    for sd in vs.get("side_data_list", []):
        if sd.get("rotation"):
            notes.append(f"rotation metadata of {sd['rotation']} degrees (filmed in a rotated orientation, usually a phone)")
            specific += 1
    return notes, specific


class VideoEncodeProbe(Probe):
    name = "video_encode"
    group = "File and metadata"
    media = (MediaType.VIDEO,)
    timeout = 60.0

    def run(self, path: Path) -> Finding:
        info = ffprobe(path)
        if not info:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="ffprobe missing or file unreadable (sudo apt install ffmpeg)")
        gop, bframes = gop_pattern(path)
        notes, specific = fingerprint(info, gop, bframes)
        if not notes:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No video stream found")
        if not specific:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="Encode details: " + "; ".join(notes) + ". Nothing in them "
                                   "points to a particular platform or device.")
        return Finding(probe=self.name, detected=True, strength=min(1.0, 0.25 + 0.2 * specific),
                       narrows=0.12, constrains=["source type", "platform"], effort=1,
                       follow_ups=follow_ups("video", "provenance"),
                       summary="; ".join(notes) + ". These describe how the file was made or "
                       "processed, not where it was filmed; platforms rewrite most of it.")
