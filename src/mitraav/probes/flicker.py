"""Lighting flicker in video: a mains-frequency (50 vs 60 Hz) hint from brightness aliasing.

Lamps flicker at twice the mains frequency (100 or 120 Hz). Sampled at the video frame rate, that
flicker folds down to a lower, predictable frequency that differs between the two cases (for
example at 30 fps: 10 Hz for 100 Hz lighting, about 0 Hz for 120 Hz). Seeing a peak at one
candidate and not the other points to a mains region. LED dimming, screens and auto-exposure can
fake or hide it, so this is a weak regional lead.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import detrend, welch

from ..core.finding import Finding, MediaType
from ..core.frames import ffprobe
from ..core.geo import grid_hint
from ..core.probe import Probe
from ..core.references import follow_ups

SECONDS = 40
MIN_PEAK_DB = 12.0


def alias(f_signal: float, fps: float) -> float:
    return abs(f_signal - round(f_signal / fps) * fps)


def analyse(brightness: np.ndarray, fps: float) -> dict | None:
    """Peak of the brightness spectrum and which mains flicker candidates it matches."""
    if len(brightness) < int(fps * 8):
        return None
    x = detrend(brightness.astype(np.float64))
    # remove slow exposure drift
    k = max(3, int(fps))
    x = x - np.convolve(x, np.ones(k) / k, mode="same")
    f, p = welch(x, fs=fps, nperseg=min(len(x), int(fps * 16)))
    keep = (f >= 0.8) & (f <= fps / 2 - 0.2)
    if keep.sum() < 8:
        return None
    ff, pp = f[keep], 10 * np.log10(p[keep] + 1e-12)
    i = int(np.argmax(pp))
    over = float(pp[i] - np.median(pp))
    peak = float(ff[i])
    cands = {50: alias(100.0, fps), 60: alias(120.0, fps)}
    tol = max(0.4, 0.04 * peak)
    match = [m for m, a in cands.items() if abs(a - peak) <= tol and a >= 0.8]
    return dict(peak=peak, over_db=over, cands=cands, match=match,
                ambiguous=abs(cands[50] - cands[60]) < 2 * tol)


def read_brightness(path: Path, fps: float, seconds: int = SECONDS) -> np.ndarray | None:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-v", "error", "-t", str(seconds), "-i", str(path), "-an",
                            "-vf", "scale=32:18,format=gray", "-f", "rawvideo", "-"],
                           capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(a) // (32 * 18)
    return a[: n * 32 * 18].reshape(n, 32 * 18).mean(axis=1) if n else None


class FlickerProbe(Probe):
    name = "lighting_flicker"
    group = "Setting and power"
    media = (MediaType.VIDEO,)
    timeout = 150.0

    def run(self, path: Path) -> Finding:
        info = ffprobe(path)
        vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if not vs:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="ffprobe missing or no video stream")
        try:
            num, den = vs.get("avg_frame_rate", "0/1").split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        if fps < 10:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Could not read a usable frame rate")
        b = read_brightness(path, fps)
        if b is None or len(b) < fps * 8:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Need at least 8 s of decodable video (is ffmpeg installed?)")
        r = analyse(b, fps)
        if not r or r["over_db"] < MIN_PEAK_DB or not r["match"]:
            msg = ("No brightness flicker that matches 100 or 120 Hz lighting at this frame rate."
                   if not r or r["over_db"] < MIN_PEAK_DB else
                   f"Brightness flickers at about {r['peak']:.1f} Hz, which neither 100 nor 120 Hz "
                   "lighting explains at this frame rate (screens, LED dimming or camera "
                   "auto-exposure can do this).")
            return Finding(probe=self.name, detected=False, summary=msg)
        if len(r["match"]) > 1 or r["ambiguous"]:
            note = (f"Brightness flickers at about {r['peak']:.1f} Hz, which both 50 Hz and 60 Hz "
                    f"mains lighting would produce at {fps:.3g} fps, so it cannot separate them")
            return Finding(probe=self.name, detected=True, strength=0.2, narrows=0.05,
                           constrains=["region"], effort=1,
                           follow_ups=follow_ups("video", "flicker"), summary=note + ".")
        m = r["match"][0]
        other = 60 if m == 50 else 50
        note = (f"Brightness flickers at about {r['peak']:.1f} Hz ({r['over_db']:.0f} dB above the "
                f"background). At {fps:.3g} fps that is what {2 * m} Hz lamp flicker produces "
                f"(a {m} Hz mains supply) and not what {2 * other} Hz would, which would fall near "
                f"{r['cands'][other]:.1f} Hz. Consistent with a **{m} Hz mains region**; LED dimming "
                "or a screen in view could imitate it")
        return Finding(probe=self.name, detected=True, meta=grid_hint(m, "lighting flicker"),
                       strength=min(1.0, 0.35 + (r["over_db"] - MIN_PEAK_DB) / 40), narrows=0.3,
                       constrains=["region"], effort=1,
                       follow_ups=follow_ups("video", "flicker"), summary=note + ".")
