"""Pasted or cloned regions in a picture: copy-move detection and error-level analysis.

* **Copy-move:** a region copied elsewhere in the same picture (to hide an object or duplicate a crowd).
  Blocks are described by their low-frequency DCT content; textured blocks with the same description and
  the same displacement vector in many places mark a clone. Repetitive patterns (tiles, windows, foliage,
  fences) can look like clones, and a clean result does not rule out rotated, scaled or retouched copies.
* **Error level (ELA) in JPEGs:** recompressing at a known quality shows how much each block still
  changes. Regions pasted from a picture with another compression history respond differently from the
  rest. Edges and texture also change the response, so the response is compared with local detail first.

Both are cheap leads for a human to look at, never a verdict.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, gaussian_filter, label

from ..core.finding import Finding, MediaType
from ..core.probe import Probe

MAX_SIDE = 1400


def _gray(im: Image.Image) -> np.ndarray:
    im = im.convert("L")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def copy_move(g: np.ndarray, min_area: int = 1800, min_shift: int = 40) -> list[dict]:
    """[{dx, dy, area, bbox}] for regions that reappear shifted by (dx, dy) elsewhere in the picture.

    The autocorrelation of the high-passed picture has a peak at the displacement of any copied region
    (its height grows with the copy's area); each candidate displacement is then checked pixel by pixel
    and only textured, matching areas of a useful size are kept."""
    h, w = g.shape
    hp = g - gaussian_filter(g, 3)
    f = np.fft.rfft2(hp, s=(2 * h, 2 * w))
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(f) ** 2, s=(2 * h, 2 * w)))
    sharp = ac - gaussian_filter(ac, 12)                         # keep only sharp peaks, not the broad texture hill
    cy, cx = h, w
    yy, xx = np.mgrid[0:2 * h, 0:2 * w]
    valid = (np.hypot(yy - cy, xx - cx) >= min_shift) & (yy >= cy)   # a shift and its negative are the same copy
    vals = sharp[valid]
    if vals.size < 100:
        return []
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-12
    work = np.where(valid, sharp, -np.inf)
    out: list[dict] = []
    blur = gaussian_filter(g, 1.0)
    detail = gaussian_filter(np.abs(hp), 4)
    for _ in range(8):
        k = int(np.argmax(work))
        if (work.flat[k] - med) / mad < 8:
            break
        py, px = divmod(k, work.shape[1])
        dy, dx = py - cy, px - cx
        work[max(0, py - 6):py + 7, max(0, px - 6):px + 7] = -np.inf
        if abs(dy) >= h or abs(dx) >= w:
            continue

        def z_at(ddy: int, ddx: int) -> float:
            yy_, xx_ = cy + ddy, cx + ddx
            if not (0 <= yy_ < 2 * h and 0 <= xx_ < 2 * w):
                return 0.0
            return float((sharp[max(0, yy_ - 2):yy_ + 3, max(0, xx_ - 2):xx_ + 3].max() - med) / mad)

        if any(z_at(int(round(dy * k)), int(round(dx * k))) >= 6 for k in (0.5, 2.0, 1 / 3, 3.0)):
            continue                                              # the pattern repeats at fractions or multiples: periodic
        a = blur[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        bb = blur[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        t = detail[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        match = (np.abs(a - bb) < np.maximum(8.0, 0.6 * t)) & (t > 5)
        match = binary_closing(binary_opening(match, iterations=2), iterations=3)
        lab, n = label(match)
        for i in range(1, n + 1):
            ys_, xs_ = np.nonzero(lab == i)
            if len(ys_) >= min_area:
                bw, bh = int(xs_.max() - xs_.min() + 1), int(ys_.max() - ys_.min() + 1)
                if np.hypot(dx, dy) < 0.9 * max(bw, bh):         # overlaps its own copy: a periodic pattern, not a paste
                    continue
                y0, x0 = ys_.min() + max(0, -dy), xs_.min() + max(0, -dx)
                out.append(dict(dx=int(dx), dy=int(dy), area=int(len(ys_)),
                                bbox=(int(x0), int(y0), int(xs_.max() + 1 + max(0, -dx)), int(ys_.max() + 1 + max(0, -dy)))))
    def iou(a: tuple, b: tuple) -> float:
        iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)

    # a periodic pattern matches itself at several shifts over the same area; a paste has one source
    keep = [v for v in out if sum(1 for o in out if o is not v and (o["dx"], o["dy"]) != (v["dx"], v["dy"])
                                  and iou(v["bbox"], o["bbox"]) > 0.3) < 2]
    return sorted(keep, key=lambda v: -v["area"])[:3]


def ela_outliers(im: Image.Image, quality: int = 90) -> list[dict]:
    """Regions whose recompression error is unusual for their amount of detail: [{bbox, z, area}]."""
    rgb = im.convert("RGB")
    s = MAX_SIDE / max(rgb.size)
    if s < 1:
        rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    re = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    err = np.abs(np.asarray(rgb, np.float64) - np.asarray(re, np.float64)).mean(axis=2)
    gy, gx = np.gradient(np.asarray(rgb.convert("L"), np.float64))
    grad = np.hypot(gx, gy)
    b = 16
    hb, wb = err.shape[0] // b, err.shape[1] // b
    e = err[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))
    gm = grad[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))
    ok = gm > 2.0
    if ok.sum() < 100:
        return []
    A = np.stack([np.ones(ok.sum()), np.log(gm[ok] + 1e-3)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.log(e[ok] + 1e-3), rcond=None)
    resid = np.zeros_like(e)
    resid[ok] = np.log(e[ok] + 1e-3) - A @ coef
    mad = np.median(np.abs(resid[ok] - np.median(resid[ok]))) * 1.4826 + 1e-6
    z = resid / mad
    mask = binary_dilation(np.abs(z) > 4.0, iterations=1) & ok
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        ys_, xs_ = np.nonzero(lab == i)
        if len(ys_) >= max(12, 0.01 * hb * wb):
            out.append(dict(bbox=(xs_.min() * b, ys_.min() * b, (xs_.max() + 1) * b, (ys_.max() + 1) * b),
                            z=float(np.mean(z[lab == i])), area=len(ys_) / (hb * wb), size=(wb * b, hb * b)))
    return sorted(out, key=lambda r: -r["area"])[:3]


class ImageTamperProbe(Probe):
    name = "image_tamper"
    group = "File and metadata"
    media = (MediaType.IMAGE,)
    frames_ok = False
    timeout = 240.0

    def run(self, path: Path) -> Finding:
        try:
            with Image.open(path) as im:
                fmt = im.format or ""
                im.load()
                g = _gray(im)
                size = im.size
                outl = ela_outliers(im) if fmt in ("JPEG", "MPO") else []
        except Exception as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=f"cannot read the picture: {e}")
        flat = float((g[:, 1:] == g[:, :-1]).mean())
        clones = copy_move(g) if flat < 0.3 else []                  # graphics and screenshots repeat by design
        notes, boxes, strength = [], [], 0.0
        for c in clones:
            x0, y0, x1, y1 = c["bbox"]
            gh, gw = g.shape
            notes.append(f"a textured area of about **{c['area']:,} px** reappears shifted by ({c['dx']:+d}, {c['dy']:+d}) px "
                         "(at analysis size) elsewhere in the picture: consistent with a region copied within the picture "
                         "(tiles, windows and foliage can imitate it)")
            boxes.append(dict(label="possible cloned area", x0=x0 / gw, y0=y0 / gh, x1=min(1.0, x1 / gw), y1=min(1.0, y1 / gh)))
            boxes.append(dict(label="its source", x0=min(1.0, (x0 + c["dx"]) / gw), y0=min(1.0, max(0.0, (y0 + c["dy"]) / gh)),
                              x1=min(1.0, (x1 + c["dx"]) / gw), y1=min(1.0, max(0.0, (y1 + c["dy"]) / gh))))
            strength = max(strength, min(0.8, 0.4 + c["area"] / 20000))
        for o in outl:
            x0, y0, x1, y1 = o["bbox"]
            w_, h_ = o["size"]
            notes.append(f"a region of about {o['area'] * 100:.0f}% of the picture responds to recompression differently from "
                         "the rest for its amount of detail (error-level analysis): consistent with content pasted from a "
                         "picture with another compression history")
            boxes.append(dict(label="different error level", x0=x0 / w_, y0=y0 / h_, x1=min(1.0, x1 / w_), y1=min(1.0, y1 / h_)))
            strength = max(strength, 0.45)
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No cloned regions" + (" or error-level outliers" if outl is not None and fmt in ("JPEG", "MPO") else "")
                           + " found. Rotated, scaled or retouched copies and careful splices are not caught, so this is not "
                             "proof of authenticity.")
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.25, effort=2, constrains=["edit history"],
                       boxes=boxes, summary=" • ".join(notes) + f" • Look at the outlined areas yourself ({size[0]}x{size[1]} picture).")
