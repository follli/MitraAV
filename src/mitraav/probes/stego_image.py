"""Hidden data in the pixels of a lossless picture: LSB replacement.

Chi-square (Westfeld and Pfitzmann) and RS analysis (Fridrich, Goljan and Du) test whether the least
significant bits of PNG, BMP or TIFF pixels have been overwritten with message bits. They find simple
LSB tools and "hide in the first N bytes" tools. They do NOT find LSB matching, steghide, adaptive
schemes (HILL, WOW, S-UNIWARD) or anything in a lossy file; a clean result here proves little.
JPEG pictures get the DCT-domain tests when the optional stego add-on is installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.stego_stats import (
    chi_square_prefix,
    lsb_plane_correlation,
    prefix_verdict,
    rs_estimate,
)

LOSSLESS = {"PNG", "BMP", "TIFF", "PPM", "PCX", "TGA"}
RS_FLAG = 0.08          # estimated share of LSBs replaced, per channel, before it counts
MAX_PIXELS = 24_000_000


def analyse_array(arr: np.ndarray) -> dict:
    """Run the pixel tests on an (h, w) or (h, w, c) uint8 array."""
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape
    out: dict = {"pixels": h * w, "channels": c}
    flat = arr.reshape(-1)
    out["prefix"], ctl = chi_square_prefix(flat, steps=20, with_control=True)
    out["verdict"] = prefix_verdict(out["prefix"], control=ctl or None)
    out["rs"] = [rs_estimate(arr[..., i]) for i in range(c)]
    out["lsb_corr"] = [lsb_plane_correlation(arr[..., i]) for i in range(c)]
    out["msb_corr"] = [lsb_plane_correlation(arr[..., i] >> 7) for i in range(c)]
    return out


def describe(res: dict, total_bytes: int) -> tuple[list[str], float]:
    """(bullets, strength) from the test results. strength is 0 when nothing stands out."""
    notes: list[str] = []
    strength = 0.0
    rs = [r for r in res["rs"] if r is not None]
    hot = [r for r in rs if r >= RS_FLAG]
    verdict = res["verdict"]
    noise_image = np.mean(res["msb_corr"]) < 0.05 and np.mean(res["lsb_corr"]) < 0.05
    if noise_image:                                      # no spatial structure at all: a noise picture, not a cover
        return [], 0.0
    if hot and len(hot) >= max(1, (len(rs) + 1) // 2):
        share = float(np.mean(hot))
        kb = share * total_bytes / 8 / 1024
        notes.append(f"RS analysis estimates that about **{share * 100:.0f}%** of the lowest bits have been replaced "
                     f"(room for roughly {kb:,.0f} KB if one bit per byte)")
        strength = max(strength, min(0.9, 0.5 + share))
    if verdict and verdict[0] == "partial":
        kb = verdict[1] * total_bytes / 8 / 1024
        notes.append(f"the chi-square test finds the lowest bits look like random message bits for the first "
                     f"**{verdict[1] * 100:.0f}%** of the pixel data and natural after that (about {kb:,.0f} KB)")
        strength = max(strength, 0.85)
    elif verdict and verdict[0] == "uniform" and not noise_image and total_bytes >= 300_000:   # small images: no power
        notes.append("the lowest bits look like random bits everywhere while the higher bits are clearly structured, "
                     "which fits a message filling the whole picture (or an image whose lowest bits were randomised, "
                     "for example by dithering or processing)")
        strength = max(strength, 0.5)
    return notes, strength


class StegoImageProbe(Probe):
    name = "hidden_data_pixels"
    group = "Hidden data"
    media = (MediaType.IMAGE,)
    frames_ok = False
    timeout = 300.0

    def run(self, path: Path) -> Finding:
        try:
            im = Image.open(path)
            fmt = im.format or ""
            im.load()
        except Exception as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=f"cannot read the picture: {e}")
        if fmt in ("JPEG", "MPO"):
            from ..core.stego_dct import dct_available
            if dct_available():
                return self._dct(path)
            return Finding(probe=self.name, detected=False, info=True,
                           summary="A JPEG is lossy, so pixel-level LSB tests do not apply (the codec rewrites those bits). "
                                   "Tests on the JPEG's own DCT coefficients need the optional stego add-on "
                                   "(Add-ons dialog, or: mitraav-setup --install stego).")
        if fmt not in LOSSLESS:
            return Finding(probe=self.name, detected=False, info=True,
                           summary=f"{fmt or 'This format'} is lossy or palette-based, so pixel LSB tests do not apply.")
        if im.mode in ("P", "1", "I;16", "I", "F") or im.width * im.height > MAX_PIXELS:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="This picture's mode or size is outside what the pixel tests handle "
                                   "(palette, 16-bit or very large images are skipped).")
        notes: list[str] = []
        alpha_note = None
        if im.mode in ("RGBA", "LA"):
            a = np.asarray(im)[..., -1]
            vals = np.unique(a)
            if len(vals) == 2 and abs(int(vals[0]) - int(vals[1])) == 1 and vals[1] == 255:
                alpha_note = ("the transparency channel varies only in its lowest bit (254 and 255) although the "
                              "picture looks opaque: a classic place to hide bits")
            elif 1 < len(vals) <= 4 and int(vals.max()) == 255 and int(vals.min()) >= 250:
                alpha_note = "the transparency channel holds near-opaque values that differ slightly (possible hidden bits)"
            im = im.convert("RGB") if im.mode == "RGBA" else im.convert("L")
        elif im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        res = analyse_array(arr)
        total_bytes = arr.size
        bullets, strength = describe(res, total_bytes)
        notes += bullets
        if alpha_note:
            notes.append(alpha_note)
            strength = max(strength, 0.6)
        if not notes:
            return Finding(
                probe=self.name, detected=False, info=True,
                summary="No sign of LSB replacement: the chi-square and RS tests look like an untouched picture. "
                        "These tests cannot see LSB matching, steghide or other adaptive methods, so this is not "
                        "proof that nothing is hidden.")
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.4, effort=2,
            constrains=["hidden content"], follow_ups=follow_ups("stego", "image_lsb"),
            summary=" • ".join(notes) + " • Consistent with LSB steganography; noise-like or heavily processed "
            "images can mimic it, so confirm with a decoder such as zsteg.")

    def _dct(self, path: Path) -> Finding:
        from ..core.stego_dct import analyse_jpeg
        return analyse_jpeg(path, self.name)
