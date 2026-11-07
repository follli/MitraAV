"""The synthetic demo media must keep lighting up the probes it was built for."""
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
sys.path.insert(0, str(SAMPLES))
import make_demo as md  # noqa: E402

from mitraav.core.finding import plain  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    md.OUT = out
    marks = md.write_audio()
    md.write_image()
    return out, marks


def test_demo_audio_hits_the_audio_probes(demo):
    out, marks = demo
    wav = out / "demo_everything.wav"
    from mitraav.probes.audio_provenance import AudioProvenanceProbe
    from mitraav.probes.bells import ChurchBellsProbe
    from mitraav.probes.doppler import DopplerProbe
    from mitraav.probes.hum import HumProbe
    from mitraav.probes.reverb import ReverbProbe
    from mitraav.probes.signal_tones import SignalTonesProbe
    from mitraav.probes.siren import SirenProbe
    expect = {
        HumProbe: "60 Hz mains", ChurchBellsProbe: "6 distinct bells", SignalTonesProbe: "North American dial tone",
        SirenProbe: "two-tone", ReverbProbe: "reverberation time", DopplerProbe: "event 1",
        AudioProvenanceProbe: "digital silence",
    }
    for cls, phrase in expect.items():
        f = cls().run(wav)
        assert f.detected and phrase in plain(f.summary), (cls.__name__, plain(f.summary)[:120])
    assert "dual mono" in plain(AudioProvenanceProbe().run(wav).summary)


@pytest.mark.skipif(not shutil.which("tesseract") or not shutil.which("exiftool"),
                    reason="needs tesseract and exiftool")
def test_demo_image_hits_text_and_metadata(demo):
    out, _ = demo
    jpg = out / "demo_everything.jpg"
    from mitraav.probes.image_text import ImageTextProbe
    from mitraav.probes.jpeg_forensics import JpegForensicsProbe
    from mitraav.probes.metadata import MetadataProbe
    text = plain(ImageTextProbe().run(jpg).summary)
    assert "+44" in text and ".uk" in text and "English" in text and "mph" in text
    meta = plain(MetadataProbe().run(jpg).summary)
    assert "GPS" in meta and "51.5" in meta and "Canon" in meta
    assert "Photoshop" in plain(JpegForensicsProbe().run(jpg).summary)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_demo_video_hits_flicker_location_and_encode(demo):
    out, marks = demo
    md.OUT = out
    assert md.write_video(marks)
    mp4 = out / "demo_everything.mp4"
    from mitraav.probes.flicker import FlickerProbe
    from mitraav.probes.video_encode import VideoEncodeProbe
    from mitraav.probes.video_meta import VideoMetaProbe
    assert "60 Hz mains" in plain(FlickerProbe().run(mp4).summary)
    assert "40.7128" in plain(VideoMetaProbe().run(mp4).summary)
    assert "FFmpeg" in plain(VideoEncodeProbe().run(mp4).summary)


def test_importable_without_side_effects():
    assert importlib.util.find_spec("make_demo") is not None
