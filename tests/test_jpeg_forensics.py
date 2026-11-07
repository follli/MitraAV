import io

import numpy as np
from PIL import Image

from mitraav.probes.jpeg_forensics import STD_LUMA, JpegForensicsProbe, libjpeg_quality


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((96, 96, 3)) * 255).astype("uint8"))


def _save(tmp_path, name="a.jpg", exif=None, **kw):
    p = tmp_path / name
    _img().save(p, "JPEG", **({"exif": exif} if exif else {}), **kw)
    return p


def test_quality_recovered():
    for q in (50, 75, 90):
        b = io.BytesIO()
        _img().save(b, "JPEG", quality=q)
        b.seek(0)
        assert libjpeg_quality(Image.open(b).quantization[0], STD_LUMA) == q


def test_standard_tables_flagged(tmp_path):
    f = JpegForensicsProbe().run(_save(tmp_path, quality=80))
    assert f.detected and "standard libjpeg tables at quality 80" in f.summary


def test_camera_tags_with_library_tables_means_resaved(tmp_path):
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Canon", "EOS 5D"
    f = JpegForensicsProbe().run(_save(tmp_path, exif=ex.tobytes(), quality=90))
    assert f.detected and "re-saved" in f.summary


def test_custom_tables_not_flagged(tmp_path):
    custom = [[max(1, (i % 7) + 2) for i in range(64)]] * 2
    f = JpegForensicsProbe().run(_save(tmp_path, qtables=custom))
    assert not f.detected and "not the standard libjpeg" in f.summary


def test_editor_software_tag(tmp_path):
    ex = Image.Exif()
    ex[0x0131] = "Adobe Photoshop 25.0"
    f = JpegForensicsProbe().run(_save(tmp_path, exif=ex.tobytes(), qtables=[[3] * 64, [3] * 64]))
    assert f.detected and "editing application" in f.summary


def test_png_is_not_applicable(tmp_path):
    p = tmp_path / "a.png"
    _img().save(p)
    f = JpegForensicsProbe().run(p)
    assert not f.detected and "Not a JPEG" in f.summary
