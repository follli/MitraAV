"""How was the camera moving? A rough read from global image motion between frames.

Phase correlation of small grey frames gives the whole-picture shift per frame. A tripod or fixed mount
shows almost none, a hand-held camera a jittery 1-10 Hz tremor, a pan or a moving vehicle a steady
drift, often with shake on top. This is context about *how* the video was shot (fixed CCTV-style,
walking, driving), not a speed or a place, and it only sees motion of the whole picture, not of objects.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..core.finding import Finding, MediaType
from ..core.probe import Probe

W, H, FPS, SECONDS = 160, 90, 10, 20


def global_shifts(frames: np.ndarray) -> np.ndarray:
    """(n-1, 2) array of (dx, dy) shifts between consecutive frames by phase correlation."""
    win = np.outer(np.hanning(frames.shape[1]), np.hanning(frames.shape[2]))
    spec = np.fft.fft2(frames * win[None])
    out = []
    for a, b in zip(spec[:-1], spec[1:]):
        r = a * np.conj(b)
        r /= np.abs(r) + 1e-9
        corr = np.fft.ifft2(r).real
        y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
        out.append((x if x < corr.shape[1] // 2 else x - corr.shape[1], y if y < corr.shape[0] // 2 else y - corr.shape[0]))
    return np.array(out, dtype=float)


def classify(shifts: np.ndarray, fps: float = FPS) -> tuple[str, dict]:
    if len(shifts) < 20:
        return "unknown", {}
    mag = np.hypot(shifts[:, 0], shifts[:, 1])
    k = max(3, int(fps))
    kernel = np.ones(k) / k
    drift = np.stack([np.convolve(shifts[:, i], kernel, mode="same") for i in range(2)], axis=1)
    jitter = np.hypot(*(shifts - drift).T)
    stats = dict(mean=float(mag.mean()), drift=float(np.hypot(*drift.mean(axis=0))), jitter=float(jitter.mean()),
                 drift_abs=float(np.hypot(*np.abs(drift).T).mean()))
    if stats["mean"] < 0.2:
        return "fixed", stats
    if stats["drift_abs"] > 0.8 and stats["jitter"] < 0.6:
        return "pan", stats
    if stats["drift_abs"] > 0.8:
        return "moving_shaky", stats
    if stats["jitter"] >= 0.35:
        return "handheld", stats
    return "fixed", stats


class CameraMotionProbe(Probe):
    name = "camera_motion"
    group = "Visual context"
    quiet = True
    media = (MediaType.VIDEO,)
    timeout = 120.0

    def run(self, path: Path) -> Finding:
        exe = shutil.which("ffmpeg")
        if not exe:
            return Finding(probe=self.name, detected=False, status="not_run", detail="ffmpeg is needed to read frames")
        try:
            p = subprocess.run([exe, "-v", "error", "-i", str(path), "-t", str(SECONDS), "-an", "-vf",
                                f"fps={FPS},scale={W}:{H},format=gray", "-f", "rawvideo", "pipe:1"],
                               capture_output=True, timeout=100)
        except subprocess.TimeoutExpired:
            return Finding(probe=self.name, detected=False, status="not_run", detail="decoding took too long")
        n = len(p.stdout) // (W * H)
        if n < 25:
            return Finding(probe=self.name, detected=False, info=True, summary="Too few frames to judge camera motion.")
        frames = np.frombuffer(p.stdout[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float64)
        kind, st = classify(global_shifts(frames))
        words = {"fixed": "a **fixed camera** (tripod, mount or CCTV-style): almost no whole-picture motion",
                 "handheld": "a **hand-held camera**: a steady tremor with little net drift",
                 "pan": "a **smooth pan or tracking move**: steady drift with little shake",
                 "moving_shaky": "a **moving camera with shake** (walking, a vehicle or a boat): drift plus tremor",
                 "unknown": ""}[kind]
        if not words:
            return Finding(probe=self.name, detected=False, info=True, summary="Too few frames to judge camera motion.")
        return Finding(probe=self.name, detected=True, strength=0.4, narrows=0.05, pill="context", effort=1,
                       summary=f"The whole-picture motion in the first {SECONDS} s is consistent with {words} "
                               f"(average shift {st['mean']:.2f} px per frame at {W}x{H}). Only camera motion is "
                               "measured, not moving objects.")
