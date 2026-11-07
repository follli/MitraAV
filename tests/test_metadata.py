from PIL import Image

from mitraav.core.references import follow_ups
from mitraav.core.runner import scan
from mitraav.probes.metadata import _to_deg


def test_dms_conversion():
    assert round(_to_deg((37, 46, 30), "N"), 4) == 37.775
    assert _to_deg((122, 25, 10), "W") < 0


def test_reference_loading():
    assert follow_ups("metadata", "gps")


def test_no_exif(tmp_path):
    p = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(p)
    f = next(x for x in scan(p) if x.probe == "metadata")
    assert f.probe == "metadata" and not f.detected


def test_camera_tags(tmp_path):
    p = tmp_path / "cam.jpg"
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "TestCam", "T1"
    Image.new("RGB", (8, 8)).save(p, exif=exif)
    f = next(x for x in scan(p) if x.probe == "metadata")
    assert f.detected and "device" in f.constrains
