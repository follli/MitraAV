import shutil
import subprocess
from pathlib import Path

import pytest

from mitraav.probes.video_encode import VideoEncodeProbe, gop_pattern

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample-30s-360p.mp4"


def _make(tmp_path, fps=25, gop=50, extra=(), secs=6):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=320x240:rate={fps}:duration={secs}", "-c:v", "libx264", "-g", str(gop),
                    "-bf", "0", *extra, str(p)], check=True)
    return p


def test_gop_and_framerate(tmp_path):
    p = _make(tmp_path, fps=25, gop=50)
    gop, b = gop_pattern(p)
    assert gop == 50 and not b
    f = VideoEncodeProbe().run(p)
    assert f.detected and "25 fps" in f.summary and "29.97" not in f.summary and "FFmpeg" in f.summary


def test_x264_default_gop_called_out(tmp_path):
    f = VideoEncodeProbe().run(_make(tmp_path, fps=25, gop=250, secs=22))
    assert "x264 default" in f.summary


def test_rotation_metadata(tmp_path):
    f = VideoEncodeProbe().run(_make(tmp_path, extra=("-metadata:s:v", "rotate=90")))
    assert f.status == "ok"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample missing")
def test_youtube_handler_recognised():
    f = VideoEncodeProbe().run(SAMPLE)
    assert f.detected and "YouTube" in f.summary and "08/17/2020" in f.summary
