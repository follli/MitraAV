"""Is this a screenshot, a screen recording frame, or a photo of a screen?

* A **screenshot** has the pixel size of a phone or monitor screen, no camera make or model, and is
  usually lossless (PNG). Names and sizes are only a hint: any image can be resized to a screen size.
* A **photo of a screen** shows moire: periodic peaks in the picture's 2-D spectrum from the screen's
  pixel grid beating with the camera's. Fabric, fences and fine patterns can fake it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.probe import Probe

SCREENS = {
    (1170, 2532), (1179, 2556), (1284, 2778), (1290, 2796), (1125, 2436), (1242, 2688), (828, 1792), (750, 1334),
    (1242, 2208), (1080, 1920), (1080, 2400), (1080, 2340), (1080, 2220), (1440, 3200), (1440, 3088), (1440, 2960),
    (720, 1600), (720, 1560), (1080, 2160), (1344, 2992), (1200, 2640),
    (1920, 1080), (2560, 1440), (3840, 2160), (1366, 768), (1440, 900), (1536, 864), (1600, 900), (1680, 1050),
    (2880, 1800), (2560, 1600), (3024, 1964), (3456, 2234), (3440, 1440), (2048, 1536), (2732, 2048), (2388, 1668),
    (2360, 1640), (1280, 720), (1280, 800), (1024, 768), (1920, 1200), (5120, 2880), (3360, 2100), (1470, 956),
}
CAMERA_TAGS = (0x010F, 0x0110)       # Make, Model


def moire_peaks(img: Image.Image) -> float:
    """Strength (0..1) of off-axis periodic peaks in the 2-D spectrum of a centre crop."""
    g = np.asarray(img.convert("L").resize((512, 512), Image.BILINEAR), dtype=np.float64)
    g = g - g.mean()
    win = np.outer(np.hanning(512), np.hanning(512))
    f = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))
    yy, xx = np.mgrid[-256:256, -256:256]
    r = np.hypot(yy, xx)
    band = (r > 40) & (r < 240)
    ring = f[band]
    if ring.size == 0:
        return 0.0
    med, top = float(np.median(ring)), np.sort(ring)[-6:]
    return float(np.clip((np.mean(top) / max(med, 1e-9) - 25) / 80, 0.0, 1.0))


class ScreenCaptureProbe(Probe):
    name = "screen_capture"
    group = "File and metadata"
    media = (MediaType.IMAGE,)
    frames_ok = False
    timeout = 30.0

    def run(self, path: Path) -> Finding:
        try:
            im = Image.open(path)
            fmt, size = im.format or "", im.size
            has_camera = False
            try:
                ex = im.getexif()
                has_camera = any(ex.get(t) for t in CAMERA_TAGS)
            except Exception:
                pass
            im.load()
        except Exception as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        w, h = size
        notes, strength = [], 0.0
        on_screen_size = (w, h) in SCREENS or (h, w) in SCREENS
        if on_screen_size and not has_camera and fmt in ("PNG", "BMP", "WEBP", "JPEG"):
            notes.append(f"the pixel size {w}x{h} matches a phone or monitor screen and there is no camera make or model: "
                         "consistent with a **screenshot**" + (" (PNG is typical)" if fmt == "PNG" else ""))
            strength = 0.5 if fmt == "PNG" else 0.35
        if has_camera or not on_screen_size:
            m = moire_peaks(im)
            if m >= 0.6:
                notes.append("the picture's spectrum has strong repeating peaks typical of **moire**, which is consistent "
                             "with a photo of a screen (fine fabric, fences and screens drawn as patterns can also do this)")
                strength = max(strength, 0.4 + 0.3 * m)
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="Nothing suggests a screenshot or a photo of a screen (size, camera tags and moire).")
        return Finding(probe=self.name, detected=True, strength=min(strength, 0.8), narrows=0.1, effort=1,
                       constrains=["origin"], summary=" • ".join(notes) + ".")
