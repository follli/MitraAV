"""Near-duplicate finder for a folder of your own files: pictures and video frames by perceptual hash,
audio by a coarse loudness-and-spectrum signature.

Useful to see which files are the same picture or recording resized, recompressed, trimmed or re-encoded
(reposts, forwarded copies, several exports of one clip). Everything is computed locally.
"""
from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .runner import media_of

HASH_BITS = 12          # of 64: images this close are usually the same picture
AUDIO_CORR = 0.96


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = 0
    for b in (g[:, :-1] > g[:, 1:]).ravel():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def image_hashes(path: Path) -> list[int]:
    try:
        with Image.open(path) as im:
            im.load()
            return [dhash(im), dhash(im.transpose(Image.FLIP_LEFT_RIGHT))]      # a mirrored repost matches the second
    except Exception:
        return []


def video_hashes(path: Path, count: int = 3) -> list[int]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return []
    out = []
    try:
        for frac_seconds in (1.0, 5.0, 15.0)[:count]:
            p = subprocess.run([exe, "-v", "error", "-ss", str(frac_seconds), "-i", str(path), "-frames:v", "1", "-vf", "scale=96:-2",
                                "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True, timeout=60)
            if p.stdout:
                out.append(dhash(Image.open(io.BytesIO(p.stdout))))
    except (subprocess.TimeoutExpired, OSError):
        return out
    return out


def audio_signature(path: Path, seconds: int = 60) -> np.ndarray | None:
    """Loudness-and-spectrum signature (per-second, 16 bands, normalised) of the first minute."""
    from .audio import AudioError, load_audio
    try:
        x, sr = load_audio(path, duration=seconds)
    except AudioError:
        return None
    n = int(sr)
    if len(x) < 3 * n:
        return None
    frames = len(x) // n
    sig = []
    for i in range(frames):
        seg = x[i * n:(i + 1) * n]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        edges = np.geomspace(60, min(8000, sr / 2 - 1), 17)
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        sig.append([spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    s = np.log10(np.array(sig) + 1e-12)
    s = s - s.mean(axis=0, keepdims=True)                      # remove overall level: a quieter copy still matches
    return s.ravel() / (np.linalg.norm(s) + 1e-12)


def find_groups(folder: Path, recursive: bool = True) -> list[list[Path]]:
    """Groups (size >= 2) of files that look like copies of each other."""
    files = [p for p in (folder.rglob("*") if recursive else folder.iterdir()) if p.is_file() and media_of(p)]
    feats: dict[Path, tuple[str, object]] = {}
    for p in sorted(files):
        m = media_of(p).value if media_of(p) else ""
        if m == "image":
            h = image_hashes(p)
            if h:
                feats[p] = ("hash", h)
        elif m == "video":
            h = video_hashes(p)
            if h:
                feats[p] = ("hash", h)
        elif m == "audio":
            s = audio_signature(p)
            if s is not None:
                feats[p] = ("audio", s)
    paths = list(feats)
    parent = {p: p for p in paths}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            ka, va = feats[a]
            kb, vb = feats[b]
            if ka != kb:
                continue
            if ka == "hash":
                close = min(hamming(x, y) for x in va for y in vb) <= HASH_BITS
            else:
                close = len(va) == len(vb) and float(np.dot(va, vb)) >= AUDIO_CORR
            if close:
                parent[find(a)] = find(b)
    groups: dict[Path, list[Path]] = {}
    for p in paths:
        groups.setdefault(find(p), []).append(p)
    return [sorted(g) for g in groups.values() if len(g) > 1]
