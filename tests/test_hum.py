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


def _hum(tmp_path, f0, secs=40, sr=4000, jump_at=None, name="h.wav"):
    import numpy as np
    from scipy.io import wavfile
    t = np.arange(sr * secs) / sr
    f = np.full_like(t, f0)
    if jump_at:
        f[int(jump_at * sr):] += 0.5
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = 0.1 * (np.sin(ph) + 0.6 * np.sin(2 * ph) + 0.4 * np.sin(3 * ph) + 0.25 * np.sin(4 * ph))
    x += 0.01 * np.random.default_rng(0).standard_normal(t.size)
    p = tmp_path / name
    wavfile.write(p, sr, (x / np.abs(x).max() * 20000).astype(np.int16))
    return p


def test_hum_at_a_sped_up_frequency_is_reported_as_a_speed_change(tmp_path):
    from mitraav.probes.hum import HumProbe
    f = HumProbe().run(_hum(tmp_path, 52.5))
    assert f.detected and "does not match 50 or 60 Hz" in f.summary and "+5.0%" in f.summary and "52.5" in f.summary


def test_an_enf_frequency_jump_is_flagged_as_a_splice(tmp_path):
    from mitraav.probes.hum import HumProbe
    f = HumProbe().run(_hum(tmp_path, 50.0, secs=60, jump_at=30))
    assert f.detected and "jumps by" in f.summary and "splice" in f.summary
    assert f.meta["grid"]["hz"] == 50


def test_grid_disagreement_between_probes_is_flagged_in_the_summary():
    from mitraav.core.finding import Finding
    from mitraav.core.geo import grid_hint
    from mitraav.core.report import location_summary
    a = Finding(probe="mains_hum", detected=True, meta=grid_hint(50, "mains hum"))
    b = Finding(probe="lighting_flicker", detected=True, meta=grid_hint(60, "lighting flicker"))
    notes = location_summary([a, b])["notes"]
    assert any("mains frequency is" in n and "50 Hz according to mains hum" in n and "60 Hz according to lighting flicker" in n for n in notes)
    assert not [n for n in location_summary([a, a.__class__(probe="x", detected=True, meta=grid_hint(50, "x"))])["notes"] if "frequency is" in n]
