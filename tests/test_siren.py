import numpy as np
from scipy.io import wavfile

from mitraav.core.finding import plain
from mitraav.probes.siren import SirenProbe

SR = 16000


def _write(tmp_path, x):
    p = tmp_path / "s.wav"
    wavfile.write(p, SR, (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16))
    return p


def _noise(secs, level=0.05, seed=0):
    return level * np.random.default_rng(seed).standard_normal(int(SR * secs))


def test_two_tone_hi_lo(tmp_path):
    secs = 20
    t = np.arange(int(SR * secs)) / SR
    f = np.where((t % 1.6) < 0.8, 650.0, 520.0)
    x = _noise(secs) + 0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    r = SirenProbe().run(_write(tmp_path, x))
    assert r.detected and "two-tone" in plain(r.summary)
    import re
    hi, lo = (int(v) for v in re.search(r"about (\d+) and (\d+) Hz", plain(r.summary)).groups())
    assert abs(hi - 650) <= 20 and abs(lo - 520) <= 20 and "wail" not in plain(r.summary)


def test_wail_sweep(tmp_path):
    secs = 30
    t = np.arange(int(SR * secs)) / SR
    f = 900 * np.exp(0.45 * np.sin(2 * np.pi * t / 6.0))          # about 575 to 1415 Hz, 6 s period
    x = _noise(secs) + 0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    r = SirenProbe().run(_write(tmp_path, x))
    assert r.detected and "wail" in plain(r.summary) and "every 6." in plain(r.summary)


def test_steady_tone_not_a_siren(tmp_path):
    t = np.arange(SR * 20) / SR
    x = _noise(20) + 0.5 * np.sin(2 * np.pi * 800 * t)
    assert not SirenProbe().run(_write(tmp_path, x)).detected


def test_noise_not_a_siren(tmp_path):
    assert not SirenProbe().run(_write(tmp_path, _noise(20, 0.3))).detected
