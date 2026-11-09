"""Does the JPEG's embedded EXIF thumbnail still match the picture?

Cameras write a small preview into the EXIF block. Editors that rewrite the picture often leave the old
preview behind, so a preview showing a different image, framing or aspect ratio is a trace of editing or
cropping after capture. Phones usually regenerate it, so a match proves little.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from .reverse_search import dhash, distance

MAX_BITS = 18           # of 64; unrelated pictures differ in about 32


def exif_thumbnail(data: bytes) -> bytes | None:
    """The JPEG preview from the EXIF (IFD1) block, or None."""
    i = data.find(b"Exif\x00\x00")
    if i < 0:
        return None
    t = data[i + 6:]
    order = {b"II": "<", b"MM": ">"}.get(t[:2])
    if not order:
        return None
    try:
        ifd0 = struct.unpack(order + "I", t[4:8])[0]
        n = struct.unpack(order + "H", t[ifd0:ifd0 + 2])[0]
        ifd1 = struct.unpack(order + "I", t[ifd0 + 2 + 12 * n:ifd0 + 6 + 12 * n])[0]
        if not ifd1:
            return None
        n1 = struct.unpack(order + "H", t[ifd1:ifd1 + 2])[0]
        off = length = None
        for k in range(n1):
            e = t[ifd1 + 2 + 12 * k:ifd1 + 14 + 12 * k]
            tag, _typ, _cnt, val = struct.unpack(order + "HHII", e)
            if tag == 0x0201:
                off = val
            elif tag == 0x0202:
                length = val
        if off is None or not length:
            return None
        return t[off:off + length]
    except (struct.error, IndexError):
        return None


class ThumbnailCheckProbe(Probe):
    name = "thumbnail_check"
    group = "File and metadata"
    media = (MediaType.IMAGE,)
    frames_ok = False
    timeout = 30.0

    def run(self, path: Path) -> Finding:
        data = Path(path).read_bytes()
        if not data.startswith(b"\xff\xd8"):
            return Finding(probe=self.name, detected=False, info=True, summary="Only JPEG files carry an EXIF preview.")
        thumb = exif_thumbnail(data[:2_000_000])
        if not thumb:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No embedded EXIF thumbnail to compare with (many phones and editors omit it).")
        try:
            t = Image.open(io.BytesIO(thumb))
            t.load()
            with Image.open(path) as main:
                main.load()
                dist = distance(dhash(t), dhash(main))
                ta, ma = t.width / t.height, main.width / main.height
        except Exception:
            return Finding(probe=self.name, detected=False, info=True, summary="The embedded thumbnail could not be read.")
        ratio = abs(ta - ma) / ma
        ratio_flip = abs(1 / ta - ma) / ma                    # preview stored rotated
        ratio = min(ratio, ratio_flip)
        notes = []
        if dist > MAX_BITS:
            notes.append(f"the embedded thumbnail looks different from the picture ({dist} of 64 hash bits differ): the "
                         "picture was probably replaced or heavily edited after the camera wrote the preview")
        if ratio > 0.04 and dist <= MAX_BITS + 6:
            notes.append(f"the thumbnail's shape ({t.width}x{t.height}) does not match the picture ({main.width}x{main.height}): "
                         "consistent with cropping after capture")
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary=f"The embedded thumbnail matches the picture ({dist} of 64 hash bits differ), so no "
                                   "sign of replacement or cropping here.")
        return Finding(probe=self.name, detected=True, strength=0.7 if dist > MAX_BITS else 0.5, narrows=0.2, effort=1,
                       constrains=["edit history"], summary=" • ".join(notes) + ".")
