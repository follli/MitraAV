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
