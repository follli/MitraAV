import numpy as np
from scipy.io import wavfile

from mitraav.probes.signal_tones import SignalTonesProbe

SR = 16000


def _wav(tmp_path, build, secs=40, noise=0.02):
    rng = np.random.default_rng(0)
    t = np.arange(SR * secs) / SR
    x = noise * rng.standard_normal(t.size)
    x += build(t)
    p = tmp_path / "t.wav"
    wavfile.write(p, SR, (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16))
    return p


def _gate(t, a, b):
    return ((t >= a) & (t < b)).astype(float)


def test_na_dial_tone(tmp_path):
    p = _wav(tmp_path, lambda t: 0.1 * (np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t))
             * _gate(t, 10, 18))
    f = SignalTonesProbe().run(p)
    assert f.detected and "North American dial tone" in f.summary and "region" in f.constrains
    assert f.regions and 5 <= f.regions[0]["t0"] <= 11


def test_na_busy_cadence(tmp_path):
    def build(t):
        on = ((t % 1.0) < 0.5) & (t > 8) & (t < 24)
        return 0.1 * (np.sin(2 * np.pi * 480 * t) + np.sin(2 * np.pi * 620 * t)) * on
    f = SignalTonesProbe().run(_wav(tmp_path, build))
    assert f.detected and "North American busy signal" in f.summary
    assert "on/off timing also fits" in f.summary


def test_eas_attention_signal(tmp_path):
    p = _wav(tmp_path, lambda t: 0.1 * (np.sin(2 * np.pi * 853 * t) + np.sin(2 * np.pi * 960 * t))
             * _gate(t, 12, 22))
    f = SignalTonesProbe().run(p)
    assert f.detected and "Emergency Alert System" in f.summary


def test_noise_only(tmp_path):
    f = SignalTonesProbe().run(_wav(tmp_path, lambda t: 0 * t, noise=0.1))
    assert not f.detected and f.status == "ok"


def test_unlisted_tone_not_matched(tmp_path):
    p = _wav(tmp_path, lambda t: 0.1 * np.sin(2 * np.pi * 1700 * t) * _gate(t, 10, 18))
    assert not SignalTonesProbe().run(p).detected


def _chime(t, f, t0, decay=0.6):
    env = np.exp(-(t - t0) / decay) * (t >= t0)
    return env * sum(a * np.sin(2 * np.pi * k * f * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3)))


def test_unlisted_chime_is_measured_with_its_pitch_order(tmp_path):
    # a struck two-note chime (G5 then E5, each with overtones): not in any table, but a measurable pattern
    p = _wav(tmp_path, lambda t: _chime(t, 784, 2.0) + _chime(t, 659, 4.0), secs=8, noise=0.003)
    f = SignalTonesProbe().run(p)
    assert f.detected and "measured" in f.summary and "G5" in f.summary and "E5" in f.summary
    assert f.summary.index("G5") < f.summary.index("E5")        # in order of sounding
    assert f.narrows <= 0.15 and f.regions


def test_repeated_beeps_are_measured(tmp_path):
    p = _wav(tmp_path, lambda t: 0.3 * np.sin(2 * np.pi * 1320 * t) * sum(_gate(t, a, a + 0.4)
                                                                         for a in (2, 3, 4, 5)), secs=8, noise=0.003)
    f = SignalTonesProbe().run(p)
    assert f.detected and "1320" in f.summary and "E6" in f.summary
