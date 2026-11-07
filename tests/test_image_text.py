import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from mitraav.probes.image_text import ImageTextProbe, clues

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="needs tesseract")


def _sign(tmp_path, lines, size=44):
    font = ImageFont.load_default(size=size)
    img = Image.new("RGB", (1600, 140 * len(lines) + 40), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((40, 30 + i * 130), t, fill="black", font=font)
    p = tmp_path / "sign.png"
    img.save(p)
    return p


def test_uk_clues(tmp_path):
    p = _sign(tmp_path, ["Call us on +44 20 7946 0958", "Visit www.example.co.uk today", "Only 5.99 pounds £5.99"])
    f = ImageTextProbe().run(p)
    assert f.detected and "+44" in f.summary and "United Kingdom" in f.summary
    assert "7946" not in f.summary and "region" in f.constrains


def test_language_from_function_words(tmp_path):
    p = _sign(tmp_path, ["the market is open from the morning", "and the street is open for all of the people"])
    f = ImageTextProbe().run(p)
    assert f.detected and "English" in f.summary


def test_plate_like_strings_not_shown(tmp_path):
    p = _sign(tmp_path, ["Parking open for the cars", "AB12CDE and the plate XY99ZZA by the road"])
    f = ImageTextProbe().run(p)
    assert "AB12CDE" not in f.summary and "XY99ZZA" not in f.summary


def test_no_text(tmp_path):
    img = Image.new("RGB", (400, 300), "gray")
    p = tmp_path / "blank.png"
    img.save(p)
    f = ImageTextProbe().run(p)
    assert not f.detected and f.status == "ok"


def test_clue_extraction_units():
    words = [(w, 90) for w in ["Speed", "limit", "30", "mph", "and", "price", "€4", "call", "+33", "1", "23", "45", "67", "89", "www.site.fr"]]
    c = clues(words)
    assert any(u == "mph" for u, _ in c["units"]) and c["phone"][0][0] == "33"
    assert c["domain"][0][0] == "fr" and any(s == "€" for s, _ in c["currency"])


def test_axis_labels_are_not_phone_numbers():
    for txt in ("+20 40 60 80 100", "+10 dB", "+12.5", "+3 4 5 6 7 8 9 10"):
        c = clues([(w, 90) for w in txt.split()])
        assert not c["phone"], txt


def test_real_phone_formats_still_found():
    for txt, code in (("+44 20 7946 0958", "44"), ("+33 1 23 45 67 89", "33"), ("+1 (555) 123-4567", "1")):
        c = clues([(w, 90) for w in txt.split()])
        assert c["phone"] and c["phone"][0][0] == code, txt
