import shutil
import subprocess

import pytest

from mitraav.core.runner import scan

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_video_gets_audio_and_container_probes(tmp_path):
    p = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=4:size=64x64:rate=5",
                    "-f", "lavfi", "-i", "sine=frequency=60:duration=4",
                    "-shortest", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(p)], check=True)
    found = {f.probe: f for f in scan(p)}
    assert found["mains_hum"].detected
    assert "video_metadata" in found
    assert "metadata" not in found   # image EXIF probe must not run on video
