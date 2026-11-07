import numpy as np
from scipy.io import wavfile

from mitraav.core.audio import iter_chunks, load_audio, sound_start, total_seconds
from mitraav.core.finding import Finding

SR = 8000


def _file(tmp_path, silence=30, sound=20):
    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(SR * silence), 0.1 * rng.standard_normal(SR * sound)])
    p = tmp_path / "s.wav"
    wavfile.write(p, SR, (x * 32767).astype(np.int16))
    return p


def test_leading_silence_skipped(tmp_path):
    p = _file(tmp_path)
    assert abs(total_seconds(p) - 50) < 0.1
    assert 24 <= sound_start(p) <= 30
    x, sr = load_audio(p)
    assert np.abs(x[sr:]).max() > 0.01 and len(x) / sr < 27


def test_chunks_start_at_sound_and_skip_silence(tmp_path):
    offs = [o for o, _, _ in iter_chunks(_file(tmp_path, silence=400, sound=20))]
    assert offs and offs[0] >= 390


def test_all_silent_yields_nothing(tmp_path):
    p = tmp_path / "z.wav"
    wavfile.write(p, SR, np.zeros(SR * 30, dtype=np.int16))
    assert list(iter_chunks(p)) == []


def test_finding_regions_roundtrip():
    f = Finding(probe="x", detected=True, regions=[dict(label="a", t0=1, t1=2, f0=3, f1=4)])
    assert f.to_dict()["regions"][0]["t1"] == 2
