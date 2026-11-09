"""Words carried by the video file itself: subtitle tracks and closed captions.

Subtitles and broadcast captions are text that does not appear in the picture, so image OCR never sees
it. This probe reads them, reports the language, how long they run and any dates, coordinates, domains
or other place clues in them (through the same text rules as the picture reader), and deliberately does
not print the text itself: it can be a private conversation.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.geo import hint, macros_for_phrase
from ..core.probe import Probe
from .image_text import STOPWORDS, clues

CUE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.]\d{3}\s*-->\s*(\d{2}):(\d{2}):(\d{2})")


def parse_srt(text: str) -> tuple[list[str], float, float, int]:
    """(words, first cue start s, last cue end s, cue count) from SubRip text."""
    cues = list(CUE.finditer(text))
    if not cues:
        return [], 0.0, 0.0, 0
    sec = lambda h, m, s: int(h) * 3600 + int(m) * 60 + int(s)  # noqa: E731
    first = sec(*cues[0].groups()[:3])
    last = sec(*cues[-1].groups()[3:])
    body = CUE.sub(" ", text)
    body = re.sub(r"<[^>]+>|\{[^}]*\}", " ", body)
    words = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\S+", body)
    return words, float(first), float(last), len(cues)


def guess_language(words: list[str]) -> str | None:
    low = [w.lower() for w in words if w.isalpha() and len(w) >= 3]
    if len(low) < 12:
        return None
    scores = {lang: sum(1 for t in low if t in sw) for lang, sw in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    second = sorted(scores.values())[-2] if len(scores) > 1 else 0
    return best if scores[best] >= 4 and scores[best] >= 2 * second else None


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


def _srt(args: list[str]) -> str:
    exe = shutil.which("ffmpeg")
    try:
        r = subprocess.run([exe, "-v", "error", *args, "-f", "srt", "pipe:1"], capture_output=True, timeout=180)
        return r.stdout.decode("utf-8", "replace")
    except (subprocess.TimeoutExpired, OSError):
        return ""


class VideoTextTracksProbe(Probe):
    name = "video_text_tracks"
    group = "Text and visuals"
    media = (MediaType.VIDEO,)
    timeout = 240.0

    def run(self, path: Path) -> Finding:
        if not shutil.which("ffmpeg"):
            return Finding(probe=self.name, detected=False, status="not_run", detail="ffmpeg is needed to read subtitle tracks")
        streams = _streams(path)
        subs = [s for s in streams if s.get("codec_type") == "subtitle"]
        has_cc = any(s.get("closed_captions") for s in streams if s.get("codec_type") == "video")
        texts: list[tuple[str, str]] = []
        for i, s in enumerate(subs):
            if s.get("codec_name") in ("dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle", "xsub"):
                texts.append((f"image-based subtitle track {i + 1} ({s.get('codec_name')})", ""))
                continue
            tag = (s.get("tags") or {}).get("language", "")
            texts.append((f"subtitle track {i + 1}" + (f" ({tag})" if tag else ""),
                          _srt(["-i", str(path), "-map", f"0:s:{i}"])))
        if has_cc:
            esc = str(path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
            texts.append(("closed captions", _srt(["-f", "lavfi", "-i", f"movie='{esc}'[out0+subcc]", "-map", "s"])))
        if not texts:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No subtitle track or closed captions in the file.")
        notes, strength, geo = [], 0.0, None
        for label, srt in texts:
            if not srt.strip():
                notes.append(f"a {label} is present (image-based or empty: nothing to read as text)")
                strength = max(strength, 0.2)
                continue
            words, t0, t1, n = parse_srt(srt)
            if not n:
                notes.append(f"a {label} is present but holds no readable cues")
                continue
            lang = guess_language(words)
            c = clues([(w, 100) for w in words])
            bit = f"a {label} holds **{n} cues** over {int(t1 - t0)} s (from {int(t0)} s), about {len(words)} words"
            if lang:
                bit += f", in **{lang}**"
            notes.append(bit)
            strength = max(strength, 0.4 + (0.1 if lang else 0.0))
            for stamp in c["timestamps"][:2]:
                notes.append(f"the text mentions the date/time **{stamp}**")
            if c["coords"]:
                lat, lon = c["coords"][0]
                notes.append(f"the text contains coordinates **{lat:.4f}, {lon:.4f}**")
                strength = 0.8
            regs = sorted({w for key in ("phone", "domain", "currency", "units") for _, w in c[key]})
            if regs:
                notes.append("place clues in the text: " + ", ".join(regs[:3]))
                geo = hint("subtitle text: " + ", ".join(regs[:2]),
                           set().union(*(macros_for_phrase(r) for r in regs)), 0.25)
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.3, effort=1,
                       constrains=["language", "date"], meta={"geo": geo} if geo else {},
                       summary=" • ".join(notes) + " • The text itself is not shown here; read it with ffmpeg "
                       "(-map 0:s:0 out.srt) if you need it.")
