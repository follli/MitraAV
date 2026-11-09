"""Steganalysis of a JPEG's DCT coefficients (needs the optional `jpeglib` package: the stego add-on).

* **JSteg-style** LSB replacement in coefficients: the chi-square attack on coefficient value pairs,
  evaluated on growing prefixes of the coefficient stream (as for pixels).
* **F5-style shrinkage:** Fridrich, Goljan and Hogea's calibration attack. Embedding by decreasing
  coefficient magnitudes (F5, nsF5-like) leaves too many zeros and too few ones; the original histogram
  is estimated by decoding, cropping four pixels and recompressing with the same quantisation, and the
  share of modified coefficients is solved for from the difference.

Not detected by these: Steghide, OutGuess (both preserve the histograms), J-UNIWARD and other adaptive
schemes, and anything embedded at a low rate. A clean result is weak evidence.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from .finding import Finding
from .references import follow_ups
from .stego_stats import chi_square_prefix, prefix_verdict

BETA_FLAG = 0.12            # clean pictures measure up to about 0.08 (phone photos are the highest)


def dct_available() -> bool:
    return importlib.util.find_spec("jpeglib") is not None


def _hist(y: np.ndarray, k: int, ell: int) -> list[float]:
    a = np.abs(y[:, :, k, ell]).ravel()
    n = max(len(a), 1)
    return [float((a == v).sum()) / n for v in (0, 1, 2)]


def calibrated_luma(path: Path, qt: np.ndarray) -> np.ndarray:
    import jpeglib
    spatial = jpeglib.read_spatial(str(path), jpeglib.JCS_GRAYSCALE).spatial
    crop = spatial[4:, 4:]
    h, w = crop.shape[0] // 8 * 8, crop.shape[1] // 8 * 8
    crop = np.ascontiguousarray(crop[:h, :w])
    out = tempfile.mktemp(suffix=".jpg")
    try:
        jpeglib.from_spatial(crop).write_spatial(out, qt=qt)
        return jpeglib.read_dct(out).Y
    finally:
        Path(out).unlink(missing_ok=True)


def shrinkage_beta(path: Path) -> float | None:
    """Estimated share of DCT coefficients changed by magnitude-decreasing embedding (F5-style)."""
    import jpeglib
    im = jpeglib.read_dct(str(path))
    y = im.Y
    if y.shape[0] * y.shape[1] < 300:
        return None
    cal = calibrated_luma(path, im.qt[0:1])
    est = []
    for k, ell in ((0, 1), (1, 0), (1, 1)):
        h, hc = _hist(y, k, ell), _hist(cal, k, ell)
        if hc[0] > 0.85:                                  # flat graphics and blank areas: too few non-zero coefficients
            continue
        den = hc[1] ** 2 + (hc[2] - hc[1]) ** 2
        if den > 0:
            est.append((hc[1] * (h[0] - hc[0]) + (h[1] - hc[1]) * (hc[2] - hc[1])) / den)
    return float(np.mean(est)) if est else None


def jsteg_verdict(path: Path) -> tuple[tuple[str, float] | None, int]:
    import jpeglib
    y = jpeglib.read_dct(str(path)).Y.reshape(-1, 64)[:, 1:].ravel()
    v = y[(y >= -64) & (y < 64)] + 64
    v = v[(v != 64) & (v != 65)]                               # JSteg skips the values 0 and 1
    pre, ctl = chi_square_prefix(v, steps=20, levels=128, with_control=True)
    return prefix_verdict(pre, control=ctl or None), len(v)


def analyse_jpeg(path: Path, probe_name: str) -> Finding:
    try:
        verdict, n = jsteg_verdict(path)
        beta = shrinkage_beta(path)
    except Exception as e:
        return Finding(probe=probe_name, detected=False, status="not_run",
                       detail=f"the JPEG could not be analysed: {type(e).__name__}: {e}")
    notes, strength = [], 0.0
    if verdict and verdict[0] == "partial":
        notes.append(f"the chi-square test on the JPEG's DCT coefficients finds the lowest bits look like random "
                     f"message bits for the first **{verdict[1] * 100:.0f}%** of the coefficient stream and natural "
                     "after that, as JSteg-style LSB embedding leaves")
        strength = 0.8
    elif verdict and verdict[0] == "uniform":
        notes.append("the lowest bits of the DCT coefficients look random throughout, which fits JSteg-style "
                     "embedding filling the whole file")
        strength = 0.55
    if beta is not None and beta >= BETA_FLAG:
        notes.append(f"calibration (decode, crop, recompress) suggests about **{beta * 100:.0f}%** of the DCT coefficients "
                     "have had their magnitude reduced, the shrinkage signature of F5-style embedding")
        strength = max(strength, min(0.9, 0.45 + beta))
    if not notes:
        return Finding(
            probe=probe_name, detected=False, info=True,
            summary="No sign of JSteg-style or F5-style embedding in the JPEG's DCT coefficients"
                    + (f" (shrinkage estimate {beta * 100:.0f}%, clean pictures read up to about 8%)" if beta is not None else "")
                    + ". These tests cannot see Steghide, OutGuess, J-UNIWARD or low-rate embedding, "
                      "so this is not proof that nothing is hidden.")
    return Finding(
        probe=probe_name, detected=True, strength=strength, narrows=0.4, effort=2, constrains=["hidden content"],
        follow_ups=follow_ups("stego", "image_dct"),
        summary=" • ".join(notes) + " • Consistent with JPEG steganography; heavy recompression or unusual "
        "encoders can also shift these statistics, so confirm with a dedicated detector.")
