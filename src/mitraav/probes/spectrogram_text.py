"""Text or pictures drawn into the spectrogram of an audio file.

A classic hidden message: the sound is shaped so that its spectrogram *spells* something (or shows a
picture), invisible to the ear but obvious in a spectrogram viewer. This probe renders the spectrogram of
successive stretches as an image and runs the text reader on it. Letters need to be legible and not
smeared by speech; pictures without letters are not found, and OCR on noise can invent words, so only
confident real-looking words count.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.spectro import compute

WINDOW_S = 24.0
IMG_WIDTHS, IMG_H = (600, 300), 360        # letters look wide at full time resolution, so try narrower pictures
MAX_WINDOWS = 24
VOWELS = set("aeiouyAEIOUY")


def render(x: np.ndarray, sr: int, width: int = 600, invert: bool = True) -> Image.Image:
    """Spectrogram as a black-and-white image (frequency up), dark text on white when invert=True."""
    n = 1024 if sr >= 32000 else 512
    f, t, db = compute(x, sr, n)
    lo, hi = np.percentile(db, [30, 99.5])
    g = (np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)[::-1]
    im = Image.fromarray(g).resize((width, IMG_H), Image.BILINEAR).filter(ImageFilter.MedianFilter(3))
    im = im.point(lambda v: 255 if v > 128 else 0)
    if invert:
        im = Image.eval(im, lambda v: 255 - v)
    return im


def read_words(im: Image.Image) -> list[tuple[str, int]]:
    exe = shutil.which("tesseract")
    if not exe:
        return []
    tmp = Path(tempfile.mkdtemp()) / "s.png"
    im.save(tmp)
    r = subprocess.run([exe, str(tmp), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=60)
    out = []
    for line in r.stdout.splitlines()[1:]:
        p = line.split("\t")
        if len(p) == 12 and p[11].strip():
            try:
                out.append((p[11].strip(), int(float(p[10]))))
            except ValueError:
                pass
    return out


def real_words(words: list[tuple[str, int]], min_conf: int = 85) -> list[str]:
    good = []
    for w, c in words:
        w = re.sub(r"^\W+|\W+$", "", w)
        if c >= min_conf and w.isalpha() and len(w) >= 4 and VOWELS & set(w) and len(set(w.lower())) >= 3:
            good.append(w)
    return good


class SpectrogramTextProbe(Probe):
    name = "spectrogram_text"
    group = "Hidden data"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 600.0

    def run(self, path: Path) -> Finding:
        if not shutil.which("tesseract"):
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Reading text in a spectrogram needs Tesseract: sudo apt install tesseract-ocr")
        hits: list[tuple[float, list[str]]] = []
        n = 0
        try:
            for off, x, sr in iter_chunks(path):
                if sr < 16000:
                    continue
                step = int(WINDOW_S * sr)
                for a in range(0, len(x), step):
                    seg = x[a:a + step]
                    if len(seg) < 3 * sr:
                        continue
                    n += 1
                    if n > MAX_WINDOWS:
                        break
                    words = []
                    for width in IMG_WIDTHS:
                        words = real_words(read_words(render(seg, sr, width)))
                        if words:
                            break
                    if words:
                        hits.append((off + a / sr, words))
                if n > MAX_WINDOWS:
                    break
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 3 s at 16 kHz or more")
        if not hits:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No text found in the spectrogram. Pictures without letters, and text too faint or "
                                   "smeared to read, would not show up.")
        shown = "; ".join(f"near {t:.0f} s: {', '.join(w[:5])}" for t, w in hits[:3])
        regions = [dict(label=f"spectrogram text near {t:.0f} s", t0=max(0.0, t - 1), t1=t + WINDOW_S, f0=0.0, f1=24000.0)
                   for t, _ in hits[:3]]
        return Finding(probe=self.name, detected=True, strength=0.75, narrows=0.4, effort=2, constrains=["hidden content"],
                       regions=regions, follow_ups=follow_ups("stego", "audio"),
                       summary=f"the text reader finds words in the **spectrogram image** ({shown}). Consistent with text "
                               "drawn into the sound; open the signal viewer on the region to look at it yourself, since OCR "
                               "on noise can invent words.")
