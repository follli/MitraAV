import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mitraav.core.runner import scan
from mitraav.probes.doppler import doppler_model, fit_doppler

SR = 16000


def _wav(path, f_of_t, seconds=24, noise=0.02):
    t = np.arange(SR * seconds) / SR
    x = 0.1 * np.sin(2 * np.pi * np.cumsum(f_of_t(t)) / SR)
    x += noise * np.random.default_rng(0).standard_normal(t.size)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


def _doppler(path):
    return next(f for f in scan(path) if f.probe == "doppler_pass")


def test_fit_recovers_parameters():
    t = np.linspace(0, 24, 200)
    popt, r2 = fit_doppler(t, doppler_model(t, 1000, 0.15, 12, 3))
    assert r2 > 0.99 and abs(popt[1] - 0.15) < 0.02


def test_true_passby(tmp_path):
    p = tmp_path / "pass.wav"
    _wav(p, lambda t: doppler_model(t, 1000, 0.15, 12, 3))
    f = _doppler(p)
    assert f.detected and "m/s" in f.summary and f.follow_ups


def test_linear_glide_is_flagged_but_hedged(tmp_path):
    p = tmp_path / "glide.wav"
    _wav(p, lambda t: 1200 - 8 * t)
    assert _doppler(p).detected


def test_rising_tone_rejected(tmp_path):
    p = tmp_path / "rise.wav"
    _wav(p, lambda t: 800 + 10 * t)
    assert not _doppler(p).detected


def test_steady_tone_rejected(tmp_path):
    p = tmp_path / "steady.wav"
    _wav(p, lambda t: np.full_like(t, 1000.0))
    assert not _doppler(p).detected


def test_noise_only(tmp_path):
    p = tmp_path / "noise.wav"
    _wav(p, lambda t: np.full_like(t, 1000.0), noise=0.05)
    x = np.random.default_rng(1).standard_normal(SR * 24) * 0.05
    wavfile.write(p, SR, (x * 32767).astype(np.int16))
    assert not _doppler(p).detected


CLIP = Path(__file__).resolve().parents[1] / "clip_sample.wav"


@pytest.mark.skipif(not CLIP.exists() or not shutil.which("ffmpeg"),
                    reason="needs clip_sample.wav and ffmpeg")
@pytest.mark.parametrize("start,lo,hi", [(1161, 105, 125), (2009, 120, 160)])
def test_real_clip_events(tmp_path, start, lo, hi):
    """Real demo audio: event A near 1278 s and the harmonic event B near 2147 s."""
    from mitraav.probes.doppler import DopplerProbe
    out = tmp_path / "w.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", "160", "-i", str(CLIP),
                    "-ac", "1", "-ar", "16000", str(out)], check=True)
    f = DopplerProbe().run(out)
    assert f.detected
    assert "event 1" in f.summary and "event 2" not in f.summary
    import re
    t0 = float(re.search(r"near (\d+) s", f.summary).group(1))
    assert lo <= t0 <= hi


def test_retarded_time_model_matches_the_exact_geometry():
    """The fitted curve must equal brute-force emission-time geometry (sound travel delay included)."""
    import numpy as np

    from mitraav.probes.doppler import C, doppler_model
    v, d, f0, tc = 95.0, 1900.0, 100.0, 40.0
    te = np.arange(-100, 200, 0.01)
    r = np.sqrt((v * (te - tc)) ** 2 + d * d)
    ta = te + r / C
    drdt = v * v * (te - tc) / r
    f_true = f0 / (1 + drdt / C)                          # approaching: drdt < 0 -> higher pitch
    t_closest_heard = tc + d / C
    sel = (ta > t_closest_heard - 40) & (ta < t_closest_heard + 40)
    f_model = doppler_model(ta[sel], f0, v / C, t_closest_heard, d / v)
    assert np.max(np.abs(f_model - f_true[sel])) < 0.02
