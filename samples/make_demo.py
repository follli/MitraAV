"""Synthetic demo media that lights up many probes at once, with known ground truth.

    python samples/make_demo.py

writes (git-ignored) into this folder:
  demo_everything.wav  about 3 minutes, stereo 44.1 kHz: 60 Hz hum throughout; a propeller fly-by
                       with a Doppler curve; a ring of six church bells; a US dial tone; a hi-lo
                       siren; a jet pass-by; six claps in a reverberant room; 4 s of digital silence
                       at the start and a 2 s gap; a 16 kHz lossy-codec cliff; identical channels.
  demo_everything.mp4  30 s video: lighting flicker consistent with 60 Hz mains, a location and date
                       tag, x264 defaults, and the propeller fly-by (Doppler event) as its soundtrack.
  demo_everything.jpg  a "photo" with GPS (London), camera tags, an editor tag, and a sign with a UK
                       phone number, web address, pound sign, mph and English words.

It is all synthetic, so it is free of third-party rights, and every real probe behaviour is exercised
on material we know the answer for. The script prints the ground truth.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import make_samples as ms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = ms.SR                       # 22050: everything is synthesised here, then resampled to 44.1 kHz
OUT = Path(__file__).parent
rng = np.random.default_rng(7)


def noise(secs: float, level: float = 0.005) -> np.ndarray:
    return level * rng.standard_normal(int(SR * secs))


def fade(x: np.ndarray, secs: float = 0.1) -> np.ndarray:
    n = min(int(SR * secs), len(x) // 2)
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def bells(secs: float = 21.0) -> np.ndarray:
    """A descending ring of six bells, a strike every 2.5 s (tuned-bell partials, ratio 0.5:1:1.2:1.5:2)."""
    x = noise(secs)
    primes = [392.0, 349.2, 311.1, 293.7, 261.6, 233.1]
    for k, prime in enumerate(primes):
        n = int(SR * 3.0)
        t = np.arange(n) / SR
        y = np.zeros(n)
        for r, a, d in [(0.5, 1.0, 3.5), (1.0, 0.9, 2.5), (1.2, 0.6, 2.0), (1.5, 0.5, 1.8),
                        (2.0, 0.7, 1.5), (2.5, 0.3, 1.0), (3.0, 0.25, 0.8)]:
            y += a * np.exp(-t / d) * np.sin(2 * np.pi * r * prime * t + rng.random() * 6.28)
        y += 0.8 * np.exp(-t / 0.04) * rng.standard_normal(n)            # clapper
        a0 = int((1.0 + 2.5 * k) * SR)
        x[a0:a0 + n] += 0.1 * y[: len(x) - a0]
    return x * (0.5 / np.abs(x).max())


def dial_tone(secs: float = 6.0) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    return noise(secs, 0.004) + 0.25 * (np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t))


def siren(secs: float = 16.0) -> np.ndarray:
    """Hi-lo two-tone siren, 650 and 520 Hz, a full cycle every 1.6 s, with overtones."""
    t = np.arange(int(SR * secs)) / SR
    f = np.where((t % 1.6) < 0.8, 650.0, 520.0)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return noise(secs, 0.01) + 0.35 * (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph))


def claps(secs: float = 24.0, rt60: float = 1.0) -> np.ndarray:
    """Six sharp sounds in a reverberant space (exponentially decaying noise impulse response)."""
    x = noise(secs, 0.002)
    ir_t = np.arange(int(SR * 1.3)) / SR
    ir = rng.standard_normal(ir_t.size) * 10 ** (-3 * ir_t / rt60)
    ir[0] = 3.0
    for k in range(6):
        a0 = int((2.0 + 3.6 * k) * SR)
        x[a0:a0 + ir.size] += 0.5 * ir[: len(x) - a0]
    return x * (0.5 / np.abs(x).max())


def build_audio() -> tuple[np.ndarray, dict]:
    jet = dict(secs=50, v=200, h=1500, d=2000, tc=25, snr=8)
    parts = [("silence", np.zeros(int(SR * 4.0))),
             ("prop fly-by", fade(ms.synth(ms.PRESETS["prop_near"]))),
             ("bells", fade(bells())),
             ("dial tone", fade(dial_tone())),
             ("siren", fade(siren())),
             ("gap", np.zeros(int(SR * 2.0))),
             ("jet pass-by", fade(ms.synth_jet(jet))),
             ("claps", fade(claps()))]
    marks, t, chunks = {}, 0.0, []
    for name, x in parts:
        marks[name] = (round(t), round(t + len(x) / SR))
        chunks.append(x)
        t += len(x) / SR
    x = np.concatenate(chunks)
    tt = np.arange(len(x)) / SR
    sound = np.ones_like(x)
    for name in ("silence", "gap"):                       # no hum inside the silent parts
        a, b = marks[name]
        sound[int(a * SR):int(b * SR)] = 0.0
    x = x + sound * sum(a * np.sin(2 * np.pi * f * tt) for f, a in ((60, 0.02), (120, 0.012), (180, 0.007)))
    x = x * (0.8 / np.abs(x).max())
    return x, marks


def to_lossy_stereo(x: np.ndarray) -> np.ndarray:
    """44.1 kHz stereo (identical channels) with an encoder-style brick wall at 16 kHz."""
    y = resample_poly(x, 2, 1)
    spec = np.fft.rfft(y)
    spec[np.fft.rfftfreq(len(y), 1 / 44100) > 16000] = 0
    y = np.fft.irfft(spec, len(y)) + 1e-4 * rng.standard_normal(len(y))
    zero = np.abs(x) < 1e-12                                # keep the digital silence digital
    z = resample_poly(zero.astype(float), 2, 1) > 0.5
    y[z[: len(y)]] = 0.0
    return np.stack([y, y], axis=1)


def write_audio() -> dict:
    x, marks = build_audio()
    st = to_lossy_stereo(x)
    wavfile.write(OUT / "demo_everything.wav", 44100, (np.clip(st, -1, 1) * 32767).astype(np.int16))
    return marks


def write_video(marks: dict) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found: skipping the video")
        return False
    start = marks["prop fly-by"][0] + 6                      # a 30 s stretch containing the Doppler event
    cmd = [ff, "-v", "error", "-y",
           "-f", "lavfi", "-i", "nullsrc=size=640x360:rate=25:duration=30,"
                                "geq=lum='128+40*sin(2*PI*5*T)':cb=128:cr=128",
           "-ss", str(start), "-t", "30", "-i", str(OUT / "demo_everything.wav"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
           "-metadata", "location=+40.7128-074.0060/", "-metadata", "creation_time=2024-06-01T12:00:00Z",
           str(OUT / "demo_everything.mp4")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("video failed:", r.stderr[-300:])
    return r.returncode == 0


def write_image() -> None:
    w, h = 1600, 1000
    base = np.zeros((h, w, 3), np.uint8)
    base[..., 0] = 70
    base[..., 1] = np.linspace(110, 170, h)[:, None]
    base[..., 2] = np.linspace(150, 90, h)[:, None]
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=44)
    lines = ["Call us on +44 20 7946 0958", "Visit www.example.co.uk today",
             "Only \u00a35.99 for the full menu", "Speed limit 30 mph",
             "The market is open for the people and the street is open from the morning",
             "Open for all of the visitors and the families from the town"]
    d.rectangle([40, 40, w - 40, 40 + 130 * len(lines)], fill=(250, 250, 245))      # one sign panel
    y = 70
    for text in lines:
        d.text((70, y), text, fill=(20, 20, 20), font=font)
        y += 130
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Canon", "EOS 5D Mark II"
    ex[0x0131] = "Adobe Photoshop 25.0"
    ex[0x0132] = "2024:06:01 12:00:00"
    sub = ex.get_ifd(0x8769)
    sub[0x9003] = "2024:06:01 11:58:30"
    gps = ex.get_ifd(0x8825)
    gps[1], gps[2] = "N", (51.0, 30.0, 2.5)                  # Westminster, London
    gps[3], gps[4] = "W", (0.0, 7.0, 28.6)
    img.save(OUT / "demo_everything.jpg", quality=90, exif=ex)


def main() -> None:
    marks = write_audio()
    ok = write_video(marks)
    write_image()
    print("wrote demo_everything.wav" + (", .mp4" if ok else "") + ", .jpg in", OUT)
    print("ground truth (seconds into the wav):")
    for name, (a, b) in marks.items():
        print(f"  {a:4d}-{b:<4d} {name}")
    print("  everywhere  60 Hz hum (and harmonics); 16 kHz brick wall; identical channels")
    print("  video: 5 Hz brightness flicker at 25 fps = 120 Hz lamp flicker = 60 Hz mains;"
          " location 40.7128 N 74.0060 W; soundtrack = the propeller fly-by (Doppler event) and hum")
    print("  image: GPS 51.5007 N 0.1246 W, Canon EOS 5D, Adobe Photoshop tag, +44 / .co.uk / pound / mph")


if __name__ == "__main__":
    main()
