import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mitraav.core.runner import scan
from mitraav.probes.doppler import DopplerProbe, doppler_model, fit_doppler

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


def _faint_pass(seed=0, secs=90, t0=45.0, beta=0.2, tau=6.0, f0=110.0, snr_amp=0.004, mirror=False):
    """A harmonic engine tone buried in noise, cut by loud broadband bursts (the ridge tracker loses it)."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR * secs) / SR
    u = (t - t0) / tau
    sign = -1 if mirror else 1
    f = f0 / (1 + sign * beta * u / np.sqrt(1 + u * u))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(snr_amp / k ** 0.5 * np.sin(k * ph) for k in range(1, 7))
    x = x + 0.02 * rng.standard_normal(t.size)
    for b in np.arange(4.0, secs - 2, 5.5):                               # wind and traffic bursts
        a = int(b * SR)
        x[a:a + int(0.9 * SR)] += 0.15 * rng.standard_normal(int(0.9 * SR))
    return x


def test_matched_filter_finds_a_faint_pass_the_tracker_loses_and_ignores_noise_and_upward_glides(tmp_path):
    from mitraav.core.combfit import detect
    from mitraav.probes.doppler import fit_doppler
    x = _faint_pass()
    found = detect(x, SR, fit_doppler)
    assert found and abs(found[0]["t0"] - 45.0) <= 4.0 and found[0]["ratio"] >= 1.6
    assert 50 < found[0]["beta"] * 343 < 90                                   # true speed 0.2 * 343 = 69 m/s
    assert detect(_faint_pass(mirror=True), SR, fit_doppler) == []             # a glide UP is not an aircraft pass
    assert detect(0.02 * np.random.default_rng(1).standard_normal(SR * 90), SR, fit_doppler) == []


def test_probe_reports_the_faint_pass_with_numbers(tmp_path):
    p = tmp_path / "faint.wav"
    wavfile.write(p, SR, (np.clip(_faint_pass(), -1, 1) * 30000).astype(np.int16))
    f = DopplerProbe().run(p)
    assert f.detected and "closest approach heard near 4" in f.summary
    assert "m/s" in f.summary


def test_stitch_joins_pieces_of_one_glide_but_not_different_harmonics():
    from mitraav.core.tracks import Track, stitch

    def piece(t0, t1, k=1.0):
        t = np.arange(t0, t1, 0.2)
        f = k * (300 - 6 * (t - 10))                                         # a steady downward glide
        return Track(t=t, f=f, prom=np.full(len(t), 12.0), level=np.full(len(t), 60.0), span=len(t))

    a, b = piece(10, 18), piece(22, 30)
    joined = stitch([a, b])
    assert len(joined) == 1 and joined[0].t[0] == 10 and joined[0].t[-1] < 30.0
    assert len(stitch([a, piece(22, 30, k=2.0)])) == 2                       # the octave above is another line
    assert len(stitch([a, piece(40, 48)])) == 2                              # a gap longer than the limit
