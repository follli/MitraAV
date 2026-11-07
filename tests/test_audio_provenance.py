import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from mitraav.probes.audio_provenance import (
    AudioProvenanceProbe,
    bandwidth_cliff,
    internal_silence,
    noise_floor_steps,
)


def _write(tmp_path, x, sr, stereo=False):
    p = tmp_path / "a.wav"
    x = np.clip(x / max(np.abs(x).max(), 1e-9) * 0.5, -1, 1)
    d = (x * 32767).astype(np.int16)
    wavfile.write(p, sr, np.stack([d, d], axis=1) if stereo else d)
    return p


def _noise(sr, secs, seed=0):
    return np.random.default_rng(seed).standard_normal(int(sr * secs))


def test_mp3_like_lowpass_found():
    sr = 44100
    spec = np.fft.rfft(_noise(sr, 20))
    spec[np.fft.rfftfreq(int(sr * 20), 1 / sr) > 16000] = 0      # encoder-style brick wall
    x = np.fft.irfft(spec) + 1e-3 * _noise(sr, 20, seed=5)[: int(sr * 20)]
    fc = bandwidth_cliff(x, sr)
    assert fc and 15500 <= fc <= 16500


def test_telephone_band_found(tmp_path):
    sr = 16000
    sos = butter(8, [300, 3400], "band", fs=sr, output="sos")
    p = _write(tmp_path, sosfilt(sos, _noise(sr, 30)), sr)
    f = AudioProvenanceProbe().run(p)
    assert f.detected and "telephone" in f.summary


def test_full_band_noise_is_clean(tmp_path):
    p = _write(tmp_path, _noise(44100, 30), 44100)
    f = AudioProvenanceProbe().run(p)
    assert not f.detected and f.status == "ok"


def test_leading_silence_dual_mono_and_clipping(tmp_path):
    sr = 22050
    x = np.r_[np.zeros(sr * 12), 3.0 * _noise(sr, 20)]
    p = tmp_path / "s.wav"
    d = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    wavfile.write(p, sr, np.stack([d, d], axis=1))
    f = AudioProvenanceProbe().run(p)
    assert f.detected and "digital silence" in f.summary and "dual mono" in f.summary
    assert "full scale" in f.summary


def test_internal_gap_and_level_step():
    sr = 8000
    x = _noise(sr, 40) * 0.2
    x[sr * 10:sr * 13] = 0.0
    assert internal_silence(x, sr)[0][0] == 10.0
    y = np.r_[_noise(sr, 20) * 0.01, _noise(sr, 20) * 0.2]
    steps = noise_floor_steps(y, sr)
    assert steps and 17 <= steps[0] <= 21
