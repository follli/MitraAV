import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mitraav.core.finding import plain
from mitraav.probes.passby import BroadbandPassbyProbe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))
import make_samples as ms  # noqa: E402


def _write(tmp_path, x, sr=ms.SR):
    p = tmp_path / "x.wav"
    wavfile.write(p, sr, (x * 32767).astype(np.int16))
    return p


def test_jet_pass_detected(tmp_path):
    p = _write(tmp_path, ms.synth_jet(ms.JETS["jet_pass"]))
    f = BroadbandPassbyProbe().run(p)
    assert f.detected and "consistent with a moving broadband source" in plain(f.summary)
    t = float(re.search(r"peaking near (\d+) s", plain(f.summary)).group(1))
    assert 52 <= t <= 62          # truth: heard CPA near 57 s
    assert "distance" in f.constrains


def test_steady_noise_not_detected(tmp_path):
    rng = np.random.default_rng(3)
    p = _write(tmp_path, 0.1 * rng.standard_normal(ms.SR * 60))
    assert not BroadbandPassbyProbe().run(p).detected


def test_short_audio_not_run(tmp_path):
    p = _write(tmp_path, np.zeros(ms.SR * 5))
    assert BroadbandPassbyProbe().run(p).status == "not_run"
