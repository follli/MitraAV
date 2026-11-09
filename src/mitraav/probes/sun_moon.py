"""Do the light and shadows fit the recorded date, time and place? Sun and moon from the metadata.

When a file records where and when it was made (GPS plus a capture time), this computes where the sun was
(height above the horizon and compass direction, hence which way and how long shadows fall) and how much
of the moon was lit. Those numbers are what to compare with the picture: shadow direction and length, the
sky, whether it is day or night. The probe only warns about a mismatch when it is blatant (a bright,
daylit picture stamped well after dark, or the reverse); everything else is left for a person to check.
The capture time's time zone is taken from the file when it says one, otherwise guessed from the
longitude, which can be off by an hour or more.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.astro import compass, moon_phase, sun_position
from ..core.finding import Finding, FollowUp, MediaType
from ..core.probe import Probe

ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _dms(v, ref) -> float | None:
    try:
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if str(ref).upper() in ("S", "W") else d
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def image_time_place(path: Path) -> tuple[datetime | None, float | None, float | None, str]:
    """(capture time with tz when known, lat, lon, how the zone was decided)."""
    with Image.open(path) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        gps = ex.get_ifd(0x8825)
    lat = _dms(gps.get(2), gps.get(1)) if gps.get(2) else None
    lon = _dms(gps.get(4), gps.get(3)) if gps.get(4) else None
    raw = sub.get(0x9003) or ex.get(0x0132)
    if not raw or lat is None or lon is None:
        return None, lat, lon, ""
    try:
        naive = datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, lat, lon, ""
    if gps.get(29) and gps.get(7):                           # GPS date and UTC time: no zone guessing needed
        try:
            h, m, s = (float(x) for x in gps[7])
            d = datetime.strptime(str(gps[29]), "%Y:%m:%d")
            return d.replace(hour=int(h), minute=int(m), second=int(s), tzinfo=UTC), lat, lon, "GPS UTC time"
        except (ValueError, TypeError):
            pass
    off = sub.get(0x9011) or sub.get(0x9010)
    mm = re.fullmatch(r"([+-])(\d{2}):(\d{2})", str(off or ""))
    if mm:
        delta = timedelta(hours=int(mm[2]), minutes=int(mm[3])) * (1 if mm[1] == "+" else -1)
        return naive.replace(tzinfo=timezone(delta)), lat, lon, "the file's own UTC offset"
    guess = timedelta(hours=round(lon / 15))
    return naive.replace(tzinfo=timezone(guess)), lat, lon, f"a guess of UTC{round(lon / 15):+d} from the longitude"


def video_time_place(path: Path) -> tuple[datetime | None, float | None, float | None, str]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None, None, None, ""
    try:
        r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=60)
        info = json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, ValueError):
        return None, None, None, ""
    tags = {str(k).lower(): str(v) for k, v in (info.get("format", {}).get("tags") or {}).items()}
    loc = next((v for k, v in tags.items() if "iso6709" in k or k.startswith("location")), None)
    m = ISO6709.match(loc.strip()) if loc else None
    when = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
    if not m or not when:
        return None, None, None, ""
    try:
        dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return None, None, None, ""
    zone = "the container time (UTC unless it says otherwise)" if dt.tzinfo else "assumed UTC"
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)), float(m[1]), float(m[2]), zone


def mean_brightness(path: Path, video: bool) -> float | None:
    try:
        if video:
            from ..core.frames import thumbnail
            t = thumbnail(path, width=320)
            if not t:
                return None
            path = t
        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((200, 200))
            return float(np.asarray(im, dtype=np.float64).mean())
    except Exception:
        return None


class SunMoonProbe(Probe):
    name = "sun_moon_check"
    group = "File and metadata"
    media = (MediaType.IMAGE, MediaType.VIDEO)
    frames_ok = False
    timeout = 60.0

    def run(self, path: Path) -> Finding:
        video = path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp", ".bmp", ".mpo")
        try:
            when, lat, lon, how = video_time_place(path) if video else image_time_place(path)
        except Exception as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=f"metadata could not be read: {e}")
        if when is None or lat is None or lon is None:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="Needs both a GPS position and a capture time in the file's metadata to compute where the "
                                   "sun was; this file does not record both.")
        elev, az = sun_position(lat, lon, when)
        lit, waxing = moon_phase(when)
        state = ("daylight" if elev > 6 else "around sunrise or sunset" if elev > -0.8 else "civil twilight" if elev > -6
                 else "nautical twilight" if elev > -12 else "night")
        local = when.strftime("%Y-%m-%d %H:%M %z")
        text = f"At **{local}** at {lat:.3f}, {lon:.3f} (time zone from {how}) the sun was about **{elev:.0f}°** above the horizon "
        text += f"toward the **{compass(az)}** ({az:.0f}°): {state}"
        if elev > 3:
            text += (f". A vertical object then casts a shadow about {1 / np.tan(np.radians(elev)):.1f}× its height, "
                     f"pointing toward the **{compass(az + 180)}**")
        text += f". The moon was about {lit * 100:.0f}% lit ({'waxing' if waxing else 'waning'})."
        b = mean_brightness(path, video)
        warn = ""
        if b is not None and elev < -12 and b > 130:
            warn = ("**The picture looks brightly lit, which does not fit a time stamp this far after dark**: check the time "
                    "zone, the clock setting, or whether the picture is flash-lit or indoors")
        elif b is not None and elev > 20 and b < 25:
            warn = ("the picture is very dark although the sun was high: it may be indoors, underexposed, or the time stamp may "
                    "be wrong")
        ups = [FollowUp("SunCalc", f"https://www.suncalc.org/#/{lat:.4f},{lon:.4f},15/{when:%Y.%m.%d}/{when:%H:%M}/1/3",
                        "Shows the sun's path and shadow directions on a map for this place and time. A third-party site.")]
        if warn:
            return Finding(probe=self.name, detected=True, strength=0.6, narrows=0.2, effort=1, constrains=["date", "time"],
                           follow_ups=ups, summary=text + " • " + warn)
        return Finding(probe=self.name, detected=False, info=True, follow_ups=ups,
                       summary=text + " Compare this with the shadows and sky in the picture; no obvious mismatch with its brightness.")
