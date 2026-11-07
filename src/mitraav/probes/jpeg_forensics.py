"""JPEG history: was this written by a camera, a standard library, or an editor?

Compares the quantization tables with the standard libjpeg tables (a library-saved file matches a
quality setting exactly; camera and editor encoders use their own tables), and reads edit traces
from the tags (Software, XMP history, ICC profile) with ExifTool when it is installed. These say how
the file was produced, not where it was taken; platform re-saves erase most traces.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32)
ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])
EDITORS = ("photoshop", "gimp", "lightroom", "snapseed", "pixelmator", "affinity", "paint.net",
           "canva", "picsart", "facetune", "luminar", "capture one", "darktable", "rawtherapee",
           "imagemagick", "pillow", "irfan")


def libjpeg_quality(table: list[int], std: np.ndarray) -> int | None:
    """The libjpeg quality (1-100) that yields exactly this table, else None."""
    t = np.asarray(table)
    if t.size != 64:
        return None
    for order in (t, t[np.argsort(ZIGZAG)]):          # natural order, or file (zigzag) order
        for q in range(1, 101):
            scale = 5000 // q if q < 50 else 200 - 2 * q
            cand = np.clip((std * scale + 50) // 100, 1, 255)
            if np.array_equal(order, cand):
                return q
    return None


def exiftool(path: Path) -> dict:
    exe = shutil.which("exiftool")
    if not exe:
        return {}
    try:
        r = subprocess.run([exe, "-j", "-a", "-G1", "-n", str(path)], capture_output=True,
                           text=True, timeout=30)
        return json.loads(r.stdout)[0]
    except Exception:
        return {}


def _tag(meta: dict, *names: str):
    for k, v in meta.items():
        if k.split(":")[-1].lower() in names:
            return v
    return None


class JpegForensicsProbe(Probe):
    name = "jpeg_forensics"
    group = "File and metadata"
    media = (MediaType.IMAGE,)
    timeout = 40.0
    frames_ok = False      # frames are re-encoded by us, which would be misleading

    def run(self, path: Path) -> Finding:
        with Image.open(path) as img:
            fmt = img.format
            q = dict(getattr(img, "quantization", {}) or {})
            sampling = JpegImagePlugin.get_sampling(img) if fmt in ("JPEG", "MPO") else None
            progressive = bool(img.info.get("progressive") or img.info.get("progression"))
        if fmt not in ("JPEG", "MPO"):          # MPO is a JPEG with extra frames (iPhones, many cameras)
            return Finding(probe=self.name, detected=False,
                           summary=f"Not a JPEG (format {fmt}); no quantization tables to compare. "
                                   "PNG and WebP are typical of screenshots, web downloads and edited exports.")
        meta = exiftool(path)
        make, model = _tag(meta, "make"), _tag(meta, "model")
        software = str(_tag(meta, "software", "creatortool") or "")
        notes, flags = [], 0
        if fmt == "MPO":
            notes.append("a **multi-picture JPEG (MPO)**, the format iPhones and several cameras write")
        ql = libjpeg_quality(q.get(0, []), STD_LUMA)
        qc = libjpeg_quality(q.get(1, []), STD_CHROMA) if 1 in q else None
        if ql is not None and (qc is not None or 1 not in q):
            notes.append(f"the quantization tables are the **standard libjpeg tables at quality {ql}**, "
                         "so it was written by software using that library (browsers, phone "
                         "gallery apps, ImageMagick, Pillow and many web services), not by a "
                         "camera's own encoder")
            flags += 1
            if make or model:
                notes.append(f"yet the tags name a camera ({make or ''} {model or ''}".strip() +
                             "), so the file was probably **re-saved** after capture")
                flags += 1
        elif q:
            notes.append("the quantization tables are **not the standard libjpeg ones**, which fits an "
                         "in-camera or vendor-specific encoder, or an editor with its own tables"
                         + (f" (luma table starts {list(q[0][:6])})" if 0 in q else ""))
        if software:
            low = software.lower()
            kind = "an editing application" if any(e in low for e in EDITORS) else "software"
            notes.append(f"the Software/CreatorTool tag names {kind}: **'{software[:60]}'**")
            flags += 1 if any(e in low for e in EDITORS) else 0
        if any(k.lower().endswith("history") or "photoshop" in k.lower() for k in meta):
            notes.append("edit history or Photoshop resources are embedded")
            flags += 1
        icc = _tag(meta, "profiledescription")
        if icc:
            notes.append(f"colour profile '{icc}'" + (" (Display P3 is typical of recent Apple devices)"
                                                       if "display p3" in str(icc).lower() else ""))
        if sampling is not None:
            notes.append({0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(sampling, "unknown")
                         + " chroma subsampling" + (", progressive" if progressive else ""))
        if not notes:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Could not read JPEG tables")
        if not flags:                      # a Software tag alone (e.g. an iOS version) is not an edit
            return Finding(probe=self.name, detected=False, info=True,
                           summary="JPEG details: " + "; ".join(notes)
                           + ". Nothing suggests editing or a re-save.")
        return Finding(probe=self.name, detected=True, strength=min(1.0, 0.25 + 0.2 * flags),
                       narrows=0.1, constrains=["source type", "edit history"], effort=1,
                       follow_ups=follow_ups("metadata", "jpeg"),
                       summary="; ".join(notes) + ". This describes how the file was produced, not "
                       "where it was taken; platforms re-save images and erase most traces.")
