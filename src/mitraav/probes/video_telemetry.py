"""Location and motion data recorded *alongside* a video: GoPro GPMF tracks, dashcam NMEA sentences and
other GPS text. These are measured by the camera's own receiver, so they are among the strongest
location clues a video can carry (and are removed by most platforms on upload).

Reported: where the track starts and ends, its length, its top speed and the UTC time, with a map link.
Nothing is decoded that the file does not already contain; DJI and other proprietary tracks are only
named, with a pointer to a tool that reads them.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

from ..core.finding import Finding, FollowUp, MediaType
from ..core.geo import hint, macros_for_latlon
from ..core.probe import Probe

NMEA = re.compile(rb"\$G[PNLA]RMC,(\d{6})(?:\.\d+)?,([AV]),(\d{2})(\d{2}\.\d+),([NS]),(\d{3})(\d{2}\.\d+),([EW]),"
                  rb"([\d.]*),([\d.]*),(\d{6})")
PROPRIETARY = {"djmd": "DJI", "dbgi": "DJI", "camm": "Google Camera Motion Metadata (phone and 360 cameras)",
               "mebx": "Apple timed metadata"}


def parse_gpmf(data: bytes) -> list[dict]:
    """GPS samples ({lat, lon, alt, speed}) from a GoPro GPMF stream (GPS5 and GPS9 blocks)."""
    out: list[dict] = []

    def walk(buf: bytes, scal: list[float] | None) -> None:
        pos, cur = 0, scal
        while pos + 8 <= len(buf):
            key = buf[pos:pos + 4]
            typ, size, repeat = buf[pos + 4], buf[pos + 5], struct.unpack(">H", buf[pos + 6:pos + 8])[0]
            length = size * repeat
            body = buf[pos + 8:pos + 8 + length]
            pos += 8 + ((length + 3) & ~3)
            if typ == 0:
                walk(body, cur)
            elif key == b"SCAL" and typ == ord("l") and size == 4:
                cur = [float(v) for v in struct.unpack(f">{len(body) // 4}i", body)]
            elif key in (b"GPS5", b"GPS9") and typ == ord("l"):
                n = size // 4
                vals = struct.unpack(f">{len(body) // 4}i", body)
                scale = (cur or [1.0])
                for i in range(repeat):
                    row = vals[i * n:(i + 1) * n]
                    sc = [scale[j] if len(scale) > j else scale[-1] for j in range(n)]
                    r = [row[j] / sc[j] for j in range(n)]
                    if abs(r[0]) <= 90 and abs(r[1]) <= 180 and (abs(r[0]) > 1e-6 or abs(r[1]) > 1e-6):
                        out.append(dict(lat=r[0], lon=r[1], alt=r[2], speed=r[3]))

    walk(data, None)
    return out


def parse_nmea(blob: bytes) -> list[dict]:
    out = []
    for m in NMEA.finditer(blob):
        if m.group(2) != b"A":
            continue
        lat = int(m.group(3)) + float(m.group(4)) / 60
        lon = int(m.group(6)) + float(m.group(7)) / 60
        lat = -lat if m.group(5) == b"S" else lat
        lon = -lon if m.group(8) == b"W" else lon
        knots = float(m.group(9) or 0)
        d, t = m.group(11).decode(), m.group(1).decode()
        yy = int(d[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy                 # NMEA has a two-digit year
        out.append(dict(lat=lat, lon=lon, speed=knots * 0.514444, time=f"{year}-{d[2:4]}-{d[0:2]} {t[0:2]}:{t[2:4]}:{t[4:6]} UTC"))
    return out


def _km(a: dict, b: dict) -> float:
    return math.hypot((a["lat"] - b["lat"]) * 111.0, (a["lon"] - b["lon"]) * 111.0 * math.cos(math.radians((a["lat"] + b["lat"]) / 2)))


def summarise_track(pts: list[dict], source: str) -> tuple[str, dict]:
    first, last = pts[0], pts[-1]
    length = sum(_km(a, b) for a, b in zip(pts, pts[1:]) if _km(a, b) < 5)
    top = max((p.get("speed") or 0.0) for p in pts)
    when = f" ({first['time']} to {last['time']})" if first.get("time") and last.get("time") else ""
    text = (f"**{source}**: {len(pts)} position samples from **{first['lat']:.5f}, {first['lon']:.5f}** to "
            f"{last['lat']:.5f}, {last['lon']:.5f}, about {length:.2f} km long, top speed about {top:.0f} m/s "
            f"({top * 3.6:.0f} km/h){when}")
    return text, first


def _streams(path: Path) -> list[dict]:
    exe = shutil.which("ffprobe")
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout or "{}").get("streams", [])
    except (subprocess.TimeoutExpired, ValueError):
        return []


def read_stream(path: Path, index: int, limit: int = 30_000_000) -> bytes:
    exe = shutil.which("ffmpeg")
    if not exe:
        return b""
    try:
        p = subprocess.Popen([exe, "-v", "error", "-i", str(path), "-map", f"0:{index}", "-c", "copy", "-f", "data", "pipe:1"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = p.stdout.read(limit)
        p.kill()
        p.wait(timeout=10)
        return data
    except (OSError, subprocess.TimeoutExpired):
        return b""


class VideoTelemetryProbe(Probe):
    name = "video_telemetry"
    group = "File and metadata"
    media = (MediaType.VIDEO,)
    timeout = 180.0

    def run(self, path: Path) -> Finding:
        tracks: list[tuple[str, list[dict]]] = []
        named: list[str] = []
        for s in _streams(path):
            tag = (s.get("codec_tag_string") or "").strip()
            if s.get("codec_type") == "data" and tag == "gpmd":
                pts = parse_gpmf(read_stream(path, s["index"]))
                if pts:
                    tracks.append(("GoPro GPS track", pts))
            elif s.get("codec_type") in ("data", "subtitle", "unknown") and tag in PROPRIETARY:
                named.append(PROPRIETARY[tag])
            elif s.get("codec_type") in ("data", "subtitle"):
                blob = read_stream(path, s["index"], 8_000_000)
                pts = parse_nmea(blob)
                if pts:
                    tracks.append(("GPS text track (NMEA)", pts))
        if not tracks:                                    # dashcams often tuck NMEA sentences into free boxes
            try:
                size = path.stat().st_size
                with open(path, "rb") as fh:
                    head = fh.read(8_000_000)
                    fh.seek(max(0, size - 8_000_000))
                    tail = fh.read()
                pts = parse_nmea(head + b"\n" + tail)
                if pts:
                    tracks.append(("NMEA GPS sentences inside the file", pts))
            except OSError:
                pass
        if not tracks and not named:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No GPS or telemetry track (GoPro, dashcam NMEA, DJI) found alongside the video. "
                                   "Platforms strip these on upload, so absence says little.")
        notes, ups, geo, first = [], [], None, None
        for src, pts in tracks:
            text, first = summarise_track(pts, src)
            notes.append(text)
        if named:
            notes.append(f"a proprietary telemetry track from {', '.join(dict.fromkeys(named))} is present but not decoded here")
        if first:
            lat, lon = first["lat"], first["lon"]
            geo = hint(f"video telemetry ({lat:.2f}, {lon:.2f})", macros_for_latlon(lat, lon), 0.9)
            ups.append(FollowUp("OpenStreetMap at the start of the track",
                                f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"))
        ups.append(FollowUp("ExifTool (-ee for embedded tracks)", "https://exiftool.org",
                            "exiftool -ee -G3 -a file lists every embedded GPS and sensor sample, including DJI tracks."))
        return Finding(probe=self.name, detected=True, strength=0.9 if tracks else 0.4, narrows=0.9 if tracks else 0.2,
                       effort=1, constrains=["location", "date"] if tracks else ["origin"], follow_ups=ups,
                       meta={"geo": geo} if geo else {},
                       summary=" • ".join(notes) + " • Recorded by the camera's own receiver, but easy to edit or spoof.")
