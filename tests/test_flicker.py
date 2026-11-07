import shutil
import subprocess

import numpy as np
import pytest

from mitraav.probes.flicker import FlickerProbe, alias, analyse


def _signal(f_alias, fps, secs=30, noise=0.3):
    t = np.arange(int(fps * secs)) / fps
    rng = np.random.default_rng(0)
    return 100 + 5 * np.sin(2 * np.pi * f_alias * t) + noise * rng.standard_normal(t.size)


def test_alias_values():
    assert abs(alias(100, 30) - 10) < 1e-9 and alias(120, 30) < 1e-9
    assert abs(alias(120, 25) - 5) < 1e-9 and alias(100, 25) < 1e-9


def test_50hz_signature_at_30fps():
    r = analyse(_signal(10.0, 30.0), 30.0)
    assert r["match"] == [50] and not r["ambiguous"]


def test_60hz_signature_at_25fps():
    r = analyse(_signal(5.0, 25.0), 25.0)
    assert r["match"] == [60]


def test_no_flicker():
    r = analyse(100 + np.random.default_rng(1).standard_normal(900), 30.0)
    assert r is None or r["over_db"] < 12 or not r["match"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_video_probe_end_to_end(tmp_path):
    p = tmp_path / "f.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=size=64x48:rate=30:duration=20,geq=lum='128+40*sin(2*PI*10*T)':cb=128:cr=128",
                    "-c:v", "libx264", "-crf", "12", str(p)], check=True)
    f = FlickerProbe().run(p)
    assert f.detected and "50 Hz" in f.summary and "region" in f.constrains
