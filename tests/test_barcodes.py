import importlib.util

import numpy as np
import pytest
from PIL import Image

from mitraav.probes.barcodes import BarcodesProbe, describe_code, gs1_country

pytestmark = pytest.mark.skipif(importlib.util.find_spec("zxingcpp") is None, reason="the codes add-on (zxing-cpp) is optional")


def _code(tmp_path, fmt, text, name="c.png", pad=40, scale=1):
    import zxingcpp
    bc = zxingcpp.create_barcode(text, getattr(zxingcpp.BarcodeFormat, fmt))
    arr = np.asarray(zxingcpp.write_barcode_to_image(bc, scale=4 * scale)) if hasattr(zxingcpp, "write_barcode_to_image") else None
    im = Image.fromarray(np.asarray(arr)).convert("L")
    canvas = Image.new("L", (im.width + 2 * pad, im.height + 2 * pad), 255)
    canvas.paste(im, (pad, pad))
    p = tmp_path / name
    canvas.save(p)
    return p


def test_gs1_prefix_gives_the_registration_country():
    assert gs1_country("4006381333931") == "Germany"
    assert gs1_country("5000112637922") == "the United Kingdom"
    assert gs1_country("9780306406157") == "a book (ISBN)"
    assert gs1_country("12345") is None


def test_sensitive_codes_are_withheld_and_urls_show_only_the_site():
    assert "withheld" in describe_code("QRCode", "WIFI:S:home;T:WPA;P:secret;;")[0] and "secret" not in describe_code("QRCode", "WIFI:S:home;T:WPA;P:secret;;")[0]
    assert "contents withheld" in describe_code("QRCode", "BEGIN:VCARD\nFN:A B\nEND:VCARD")[0]
    text, country, _g, _f = describe_code("QRCode", "https://example.co.uk/a/b?token=abc123")
    assert "example.co.uk" in text and "abc123" not in text and country == "the United Kingdom"
    text, _c, geo, fu = describe_code("QRCode", "geo:48.8584,2.2945")
    assert "48.85840" in text and geo["any_of"] == ["europe"] and fu


def test_qr_code_in_a_picture_is_read_with_a_box_and_a_geo_hint(tmp_path):
    p = _code(tmp_path, "QRCode", "geo:35.6586,139.7454")
    f = BarcodesProbe().run(p)
    assert f.detected and "map location" in f.summary and f.boxes and f.meta["geo"]["any_of"] == ["asia"]
    assert 0 < f.boxes[0]["x0"] < f.boxes[0]["x1"] <= 1


def test_product_barcode_gives_a_country_hint_and_a_blank_picture_gives_none(tmp_path):
    p = _code(tmp_path, "EAN13", "4006381333931", "e.png")
    f = BarcodesProbe().run(p)
    assert f.detected and "Germany" in f.summary and f.meta["geo"]["any_of"] == ["europe"]
    blank = tmp_path / "b.png"
    Image.new("L", (200, 200), 255).save(blank)
    assert not BarcodesProbe().run(blank).detected
