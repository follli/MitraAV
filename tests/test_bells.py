import numpy as np
from scipy.io import wavfile

from mitraav.core.finding import plain
from mitraav.probes.bells import ChurchBellsProbe, note_name

SR = 22050


def _bell(t0, prime, secs, rng):
    t = np.arange(int(SR * secs)) / SR
    y = np.zeros_like(t)
    for r, a, d in [(0.5, 1.0, 3.5), (1.0, 0.9, 2.5), (1.2, 0.6, 2.0), (1.5, 0.5, 1.8),
                    (2.0, 0.7, 1.5), (2.5, 0.3, 1.0), (3.0, 0.25, 0.8)]:
        y += a * np.exp(-t / d) * np.sin(2 * np.pi * r * prime * t + rng.random() * 6.28)
    y += 0.8 * np.exp(-t / 0.04) * rng.standard_normal(t.size)       # clapper transient
    return int(t0 * SR), y


def _file(tmp_path, primes, gap=3.0, noise=0.005, secs=None):
    rng = np.random.default_rng(1)
    secs = secs or (len(primes) * gap + 6)
    x = noise * rng.standard_normal(int(SR * secs))
    for i, p in enumerate(primes):
        a, y = _bell(2 + i * gap, p, 3.0, rng)
        x[a:a + len(y)] += 0.1 * y[: len(x) - a]
    p = tmp_path / "b.wav"
    wavfile.write(p, SR, (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16))
    return p


def test_ring_of_bells_detected(tmp_path):
    primes = [392.0, 349.2, 311.1, 293.7, 261.6, 233.1]       # descending scale; nominal = 2x
    f = ChurchBellsProbe().run(_file(tmp_path, primes))
    assert f.detected and "region" in f.constrains and f.follow_ups
    assert "6 distinct bells" in plain(f.summary)
    assert "nominal about 466 Hz" in plain(f.summary) and "nominal about 784 Hz" in plain(f.summary)


def test_repeated_single_bell(tmp_path):
    f = ChurchBellsProbe().run(_file(tmp_path, [330.0] * 4))
    assert f.detected and "1 distinct bell" in f.summary


def test_noise_not_detected(tmp_path):
    rng = np.random.default_rng(2)
    p = tmp_path / "n.wav"
    wavfile.write(p, SR, (0.1 * rng.standard_normal(SR * 30) * 32767).astype(np.int16))
    assert not ChurchBellsProbe().run(p).detected


def test_plain_tones_not_bells(tmp_path):
    t = np.arange(SR * 20) / SR
    x = 0.1 * (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)) * (t % 3 < 1)
    p = tmp_path / "t.wav"
    wavfile.write(p, SR, (x * 32767).astype(np.int16))
    assert not ChurchBellsProbe().run(p).detected


def test_note_name():
    assert note_name(440.0).startswith("A4") and note_name(261.63).startswith("C4")
