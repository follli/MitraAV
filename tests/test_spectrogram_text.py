import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import istft

from mitraav.probes.spectrogram_text import SpectrogramTextProbe

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="needs tesseract")
SR = 32000


def drawn_text_audio(text="HELLO WORLD", secs=12, seed=0):
    """Audio whose spectrogram spells `text` (random-phase inverse STFT of a text image)."""
    n = 1024
    frames = int(secs * SR / (n // 2))
    img = Image.new("L", (frames, n // 2 + 1), 0)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=130)
    except TypeError:
        font = ImageFont.load_default()
    d.text((20, 40), text, fill=255, font=font)
    mag = np.asarray(img, dtype=float)[::-1] / 255.0                       # frequency up
    rng = np.random.default_rng(seed)
    Z = mag * np.exp(2j * np.pi * rng.random(mag.shape)) * 200
    _, x = istft(Z, fs=SR, nperseg=n, noverlap=n // 2)
    x = x / np.abs(x).max() * 0.5
    return x + 0.002 * rng.standard_normal(len(x))


def test_words_drawn_into_the_spectrogram_are_found(tmp_path):
    x = drawn_text_audio()
    p = tmp_path / "msg.wav"
    wavfile.write(p, SR, (x * 30000).astype(np.int16))
    f = SpectrogramTextProbe().run(p)
    assert f.detected and ("HELLO" in f.summary.upper() or "WORLD" in f.summary.upper()) and f.regions


def test_ordinary_audio_gives_no_spectrogram_text(tmp_path):
    t = np.arange(SR * 10) / SR
    rng = np.random.default_rng(0)
    x = 0.2 * np.sin(2 * np.pi * (300 + 40 * np.sin(2 * np.pi * 0.7 * t)) * t) + 0.05 * rng.standard_normal(len(t))
    p = tmp_path / "plain.wav"
    wavfile.write(p, SR, (x * 30000).astype(np.int16))
    assert not SpectrogramTextProbe().run(p).detected
