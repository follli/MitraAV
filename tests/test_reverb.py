import numpy as np
from scipy.io import wavfile

from mitraav.core.finding import plain
from mitraav.probes.reverb import ReverbProbe

SR = 16000


def _claps(tmp_path, rt60, n=6, secs=30, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.002 * rng.standard_normal(SR * secs)
    ir_t = np.arange(int(SR * min(rt60 * 1.3, 3.0))) / SR
    ir = rng.standard_normal(ir_t.size) * 10 ** (-3 * ir_t / rt60)     # -60 dB at rt60
    ir[0] = 3.0
    for k in range(n):
        a = int((3 + k * 4.5) * SR)
        imp = np.zeros(SR)
        imp[0] = 1.0
        y = np.convolve(imp, ir)[: SR * 3]
        x[a:a + len(y)] += 0.5 * y
    p = tmp_path / "c.wav"
    wavfile.write(p, SR, (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16))
    return p


def test_long_decay(tmp_path):
    f = ReverbProbe().run(_claps(tmp_path, 1.4))
    assert f.detected and "about 1." in plain(f.summary) and "enclosed" in f.summary


def test_dry_decay(tmp_path):
    f = ReverbProbe().run(_claps(tmp_path, 0.15))
    assert f.detected and "short" in f.summary


def test_no_impulses(tmp_path):
    p = tmp_path / "n.wav"
    wavfile.write(p, SR, (0.1 * np.random.default_rng(1).standard_normal(SR * 20) * 32767).astype(np.int16))
    assert not ReverbProbe().run(p).detected
