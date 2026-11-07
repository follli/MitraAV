"""Reverse image search helper: the file to upload, a perceptual hash, and where to upload it.

A specific building, view or photo is usually named fastest by a reverse image search, which this
tool never does itself (it makes no network calls). For a video it points at an extracted frame.
The dHash (64 bits) lets you recognise the same picture later: hashes that differ in about 10 bits
or fewer are usually the same image, resized or recompressed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = 0
    for b in (g[:, :-1] > g[:, 1:]).ravel():
        bits = (bits << 1) | int(b)
    return bits


def distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class ReverseSearchProbe(Probe):
    name = "reverse_search"
    group = "Visual context"
    quiet = True
    media = (MediaType.IMAGE,)
    timeout = 20.0

    def run(self, path: Path) -> Finding:
        with Image.open(path) as img:
            w, h = img.size
            h64 = dhash(img)
        note = ("a reverse image search is usually the quickest way to name a specific place or "
                f"building. Upload this file yourself: **{Path(path).resolve()}** ({w}x{h}). "
                f"Perceptual hash **{h64:016x}**: other copies whose hash differs by about 10 bits or "
                "fewer are usually the same picture, resized or recompressed")
        if min(w, h) < 200:
            note += ". The picture is small, so matches will be unreliable"
        return Finding(probe=self.name, detected=True, strength=0.5, narrows=0.05, pill="hash created",
                       constrains=["place"], effort=1, follow_ups=follow_ups("visual", "reverse"),
                       summary=note[0].upper() + note[1:] + ".")
