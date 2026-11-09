"""What a file's name and timestamps say about where it came from.

Messaging apps, phones, action cameras, dashcams and downloaders name files in recognisable ways, and
some names carry a date, a time or a video id. These are clues about the *route* a file took (and often
about a re-save date, not the capture date), so each is worded as such. Names are easy to change.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from ..core.finding import Finding, FollowUp, MediaType
from ..core.probe import Probe

# (regex, description, kind) ; kind: "route" = saved/received date, "capture" = taken date, "edit", "id"
PATTERNS = [
    (r"^(IMG|VID|PTT|AUD|DOC|STK)-(\d{8})-WA\d{4}", "a WhatsApp file (the date is when it was received or saved, not taken)", "route", 2),
    (r"^PXL_(\d{8})_(\d{6})\d*", "a Google Pixel camera file", "capture", 1),
    (r"^(?:IMG|VID|MVIMG)_(\d{8})_(\d{6})", "an Android camera file", "capture", 1),
    (r"^(\d{8})_(\d{6})(?:\D|$)", "an Android or generic camera file name (date and time)", "capture", 1),
    (r"^IMG_E\d{4}\b", "an iPhone picture that was edited in Photos (the E means an edited copy)", "edit", None),
    (r"^IMG_\d{4}\b", "an iPhone, Canon or similar camera numbering", "id", None),
    (r"^Screenshot[_ ](\d{4}-\d{2}-\d{2})[ _]at[ _](\d{2})\.(\d{2})\.(\d{2})", "a macOS screenshot", "capture", 1),
    (r"^Screenshot_(\d{8})-(\d{6})", "an Android screenshot", "capture", 1),
    (r"^Screen ?[Ss]hot\b", "a screenshot", "id", None),
    (r"^photo_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", "a Telegram download (the time is in the sender's or your time zone)", "route", 1),
    (r"^(?:video|file)_\d+", "a Telegram download", "id", None),
    (r"^signal-(\d{4}-\d{2}-\d{2})-(\d{6})", "a Signal attachment", "route", 1),
    (r"^FB_IMG_(\d{13})", "a Facebook download (the number is a millisecond timestamp)", "epoch", 1),
    (r"^Snapchat-\d+", "a Snapchat save", "id", None),
    (r"^DJI_(\d{14})_\d+", "a DJI drone video", "capture", 1),
    (r"^DJI_\d{4}\b", "a DJI drone file", "id", None),
    (r"^G(?:OPR|X\d{2}|H\d{2}|L\d{2})\d{4}", "a GoPro file", "id", None),
    (r"^DSC[FN_]?\d{4,5}\b", "a Nikon, Sony or Fujifilm camera numbering", "id", None),
    (r"^FILE(\d{6})-(\d{6})", "a dashcam file (YYMMDD-HHMMSS)", "capture", 1),
    (r"^(\d{4})_(\d{4})_(\d{6})_\d+", "a dashcam file (year, month-day, time)", "capture", None),
    (r"^zoom_\d+", "a Zoom recording", "id", None),
    (r"^(\d{4}-\d{2}-\d{2})[ _T](\d{2})[-.:](\d{2})[-.:](\d{2})", "a date-and-time file name (screen recorder or export)", "capture", 1),
]
YOUTUBE_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def _date_from(m: re.Match, idx: int | None, kind: str) -> str:
    if idx is None:
        return ""
    g = m.group(idx)
    try:
        if kind == "epoch":
            return datetime.fromtimestamp(int(g) / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        if len(g) == 8 and g.isdigit():
            return f"{g[:4]}-{g[4:6]}-{g[6:8]}"
        if len(g) == 14 and g.isdigit():
            return f"{g[:4]}-{g[4:6]}-{g[6:8]} {g[8:10]}:{g[10:12]}:{g[12:14]}"
        if len(g) == 6 and g.isdigit():
            return f"20{g[:2]}-{g[2:4]}-{g[4:6]}"
        return g
    except (ValueError, OverflowError, OSError):
        return ""


def clues_for(name: str) -> list[tuple[str, str, str]]:
    """[(description, date text or '', kind)] for a file name (extension ignored)."""
    stem = Path(name).stem
    out = []
    for rx, desc, kind, idx in PATTERNS:
        m = re.match(rx, stem)
        if m:
            out.append((desc, _date_from(m, idx, kind), kind))
            break                                           # the most specific pattern wins
    return out


class FilenameCluesProbe(Probe):
    name = "filename_clues"
    group = "File and metadata"
    media = (MediaType.ANY,)
    timeout = 10.0

    def run(self, path: Path) -> Finding:
        p = Path(path)
        notes, ups = [], []
        for desc, date, kind in clues_for(p.name):
            if kind == "id" and not date:
                notes.append(f"the name is consistent with {desc}")
            else:
                notes.append(f"the name is consistent with {desc}" + (f", dated **{date}**" if date else ""))
        m = YOUTUBE_ID.search(p.stem)
        if m:
            notes.append(f"the name ends in a YouTube-style video id (**{m.group(1)}**), which suggests the file was "
                         "downloaded from there")
            ups.append(FollowUp("Open the YouTube page for this id", f"https://www.youtube.com/watch?v={m.group(1)}",
                                "A page on a third-party site; it may log your visit."))
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            mtime = ""
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="The file name does not match a known camera, messaging-app or downloader pattern"
                                   + (f" (file last modified {mtime}, which says when it was saved here)" if mtime else "") + ".")
        if mtime:
            notes.append(f"the file was last modified {mtime} on this computer (a copy date, not a capture date)")
        return Finding(probe=self.name, detected=True, strength=0.3, narrows=0.15, effort=1, constrains=["date", "route"],
                       follow_ups=ups, summary=" • ".join(notes) + " • File names are easy to change.")
