import numpy as np
from scipy.io import wavfile

from mitraav.core.runner import scan
from mitraav.core.spectro import compute, to_rgb

SR = 8000


def _make(path, hz=None, seconds=20):
    rng = np.random.default_rng(0)
    t = np.arange(SR * seconds) / SR
    x = 0.05 * rng.standard_normal(t.size)
    if hz:
        x += 0.02 * np.sin(2 * np.pi * hz * t) + 0.015 * np.sin(2 * np.pi * 2 * hz * t)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


def test_detects_50(tmp_path):
    p = tmp_path / "a.wav"
    _make(p, 50)
    f = next(x for x in scan(p) if x.probe == "mains_hum")
    assert f.detected and "50 Hz mains" in f.summary and f.follow_ups


def test_detects_60(tmp_path):
    p = tmp_path / "b.wav"
    _make(p, 60)
    f = next(x for x in scan(p) if x.probe == "mains_hum")
    assert f.detected and "60 Hz mains" in f.summary


def test_noise_only(tmp_path):
    p = tmp_path / "c.wav"
    _make(p, None)
    f = next(x for x in scan(p) if x.probe == "mains_hum")
    assert not f.detected and f.status == "ok"


def test_spectrogram_peak_and_image():
    t = np.arange(SR * 2) / SR
    freqs, _, db = compute(np.sin(2 * np.pi * 1000 * t).astype(np.float32), SR)
    assert abs(freqs[np.argmax(db.mean(axis=1))] - 1000) < 10
    rgb = to_rgb(db)
    assert rgb.dtype == np.uint8 and rgb.shape[:2] == db.shape and rgb.shape[2] == 3


def test_enf_track_follows_a_drifting_supply(tmp_path):
    import numpy as np
    from scipy.io import wavfile

    from mitraav.probes.hum import HumProbe, enf_track
    sr, secs = 4000, 60
    t = np.arange(sr * secs) / sr
    f = 50 + 0.05 * np.sin(2 * np.pi * t / 40)                 # a slow +/-50 mHz wander
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = 0.1 * (np.sin(ph) + 0.6 * np.sin(2 * ph) + 0.4 * np.sin(3 * ph)) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    tr = enf_track(x, sr, 50, [50, 100, 150])
    assert abs(np.nanmin(tr) - 49.95) < 0.02 and abs(np.nanmax(tr) - 50.05) < 0.02
    p = tmp_path / "h.wav"
    wavfile.write(p, sr, (x / np.abs(x).max() * 20000).astype(np.int16))
    fnd = HumProbe().run(p)
    assert fnd.detected and "frequency track runs from 49.9" in fnd.summary
