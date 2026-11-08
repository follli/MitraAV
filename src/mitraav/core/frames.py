from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

N_FRAMES = 5


def ffprobe(path: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format",
                            "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def extract_frames(path: Path, n: int = N_FRAMES) -> list[tuple[float, Path]]:
    """Evenly spaced JPEG frames as (seconds, file), cached in the temp dir."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return []
    try:
        dur = float(ffprobe(path).get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        dur = 0.0
    times = [dur * (i + 0.5) / n for i in range(n)] if dur > 0 else [0.0]
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    d = Path(tempfile.gettempdir()) / "mitraav-cache" / f"frames-{key}"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i, t in enumerate(times):
        f = d / f"f{i}.jpg"
        if not f.exists():
            try:
                subprocess.run([exe, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
                                "-frames:v", "1", "-q:v", "2", str(f)],
                               capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                continue
        if f.exists() and f.stat().st_size > 0:
            out.append((t, f))
    return out


def frame_at(path: Path, seconds: float) -> Path | None:
    """The full-size video frame at `seconds` as a JPEG, cached in the temp dir (None on failure)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{int(seconds * 1000)}".encode()).hexdigest()[:16]
    d = Path(tempfile.gettempdir()) / "mitraav-cache"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"frame-{key}.jpg"
    if not (out.exists() and out.stat().st_size > 0):
        try:
            subprocess.run([exe, "-v", "error", "-y", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(path),
                            "-frames:v", "1", "-q:v", "2", str(out)], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
    return out if out.exists() and out.stat().st_size > 0 else None


def thumbnail(path: Path, width: int = 480) -> Path | None:
    """One scaled JPEG frame from a video (about 1 s in, or the first frame of a very short clip),
    cached in the temp dir. None if ffmpeg is missing or the file has no decodable video."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{width}".encode()).hexdigest()[:16]
    d = Path(tempfile.gettempdir()) / "mitraav-cache"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"thumb-{key}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    for t in ("1", "0"):
        try:
            subprocess.run([exe, "-v", "error", "-y", "-ss", t, "-i", str(path), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", str(out)],
                           capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        if out.exists() and out.stat().st_size > 0:
            return out
    return None
