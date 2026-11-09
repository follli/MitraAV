import io
import struct

import numpy as np
from PIL import Image

from mitraav.probes.filename_clues import FilenameCluesProbe, clues_for
from mitraav.probes.image_text import DATE_RX, clues, parse_coords
from mitraav.probes.provenance_ai import ProvenanceAiProbe
from mitraav.probes.screen_capture import ScreenCaptureProbe, moire_peaks
from mitraav.probes.thumbnail_check import ThumbnailCheckProbe, exif_thumbnail


def _png_with_text(tmp_path, key, value, name="a.png"):
    img = Image.new("RGB", (32, 32), "gray")
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text(key, value)
    p = tmp_path / name
    img.save(p, pnginfo=info)
    return p


def test_stable_diffusion_png_parameters_are_read(tmp_path):
    val = "a red fox in snow, detailed\nNegative prompt: blurry\nSteps: 28, Sampler: Euler a, CFG scale: 7, Seed: 5, Size: 512x768, Model: dreamshaper"
    f = ProvenanceAiProbe().run(_png_with_text(tmp_path, "parameters", val))
    assert f.detected and f.strength >= 0.9 and "AI-generated" in f.summary
    assert "Sampler Euler a" in f.summary and "a red fox in snow" in f.summary


def test_c2pa_xmp_source_type_and_editor_names_are_found(tmp_path):
    img = Image.new("RGB", (16, 16), "white")
    p = tmp_path / "x.jpg"
    img.save(p)
    j = p.read_bytes()
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia" '
           b'xmp:CreatorTool="Adobe Firefly"/></x:xmpmeta>')
    seg = b"\xff\xe1" + struct.pack(">H", 2 + 29 + len(xmp)) + b"http://ns.adobe.com/xap/1.0/\x00" + xmp
    c2pa = b"\xff\xeb" + struct.pack(">H", 2 + 40) + b"JP\x00\x00jumb\x00\x00\x00\x00c2pa.claim_generator\x00\x00GPT Image 1"
    (tmp_path / "ai.jpg").write_bytes(j[:2] + seg + c2pa + j[2:])
    f = ProvenanceAiProbe().run(tmp_path / "ai.jpg")
    assert f.detected and "C2PA" in f.summary and "AI-generated" in f.summary and "Firefly" in f.summary
    assert any("contentcredentials" in u.url for u in f.follow_ups)


def test_a_plain_file_has_no_provenance_lead(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(p)
    f = ProvenanceAiProbe().run(p)
    assert not f.detected and f.info


def test_filename_patterns_give_platform_and_date_and_a_video_id():
    d = {n: clues_for(n) for n in ["IMG-20240601-WA0003.jpg", "PXL_20240601_123456789.jpg", "FB_IMG_1717243200000.jpg",
                                   "Screenshot 2024-06-01 at 12.34.56.png", "photo_2024-06-01_12-00-00.jpg", "IMG_E1234.JPG"]}
    assert d["IMG-20240601-WA0003.jpg"][0][1] == "2024-06-01" and "received" in d["IMG-20240601-WA0003.jpg"][0][0]
    assert d["PXL_20240601_123456789.jpg"][0][1] == "2024-06-01"
    assert d["FB_IMG_1717243200000.jpg"][0][1].startswith("2024-06-01")
    assert "edited" in d["IMG_E1234.JPG"][0][0]


def test_filename_probe_links_a_youtube_id(tmp_path):
    p = tmp_path / "Some title [dQw4w9WgXcQ].mp4"
    p.write_bytes(b"x")
    f = FilenameCluesProbe().run(p)
    assert f.detected and any("dQw4w9WgXcQ" in u.url for u in f.follow_ups)
    q = tmp_path / "random.mp4"
    q.write_bytes(b"x")
    assert not FilenameCluesProbe().run(q).detected


def _jpeg_with_thumbnail(tmp_path, main: Image.Image, thumb: Image.Image, name):
    buf = io.BytesIO()
    main.save(buf, "JPEG", quality=90)
    tb = io.BytesIO()
    thumb.save(tb, "JPEG")
    t = tb.getvalue()
    # TIFF: header, IFD0 (0 entries) -> IFD1 with offset+length of the thumbnail
    ifd0_off = 8
    ifd1_off = ifd0_off + 2 + 4
    ifd1 = struct.pack(">H", 2) + struct.pack(">HHII", 0x0201, 4, 1, ifd1_off + 2 + 24 + 4) \
        + struct.pack(">HHII", 0x0202, 4, 1, len(t)) + struct.pack(">I", 0)
    tiff = b"MM\x00*" + struct.pack(">I", ifd0_off) + struct.pack(">H", 0) + struct.pack(">I", ifd1_off) + ifd1 + t
    app1 = b"\xff\xe1" + struct.pack(">H", 2 + 6 + len(tiff)) + b"Exif\x00\x00" + tiff
    j = buf.getvalue()
    p = tmp_path / name
    p.write_bytes(j[:2] + app1 + j[2:])
    return p


def test_embedded_thumbnail_mismatch_and_cropping_are_found(tmp_path):
    rng = np.random.default_rng(0)
    a = Image.fromarray((rng.random((60, 80, 3)) * 255).astype(np.uint8)).resize((400, 300))
    b = Image.fromarray((rng.random((60, 80, 3)) * 255).astype(np.uint8)).resize((400, 300))
    same = _jpeg_with_thumbnail(tmp_path, a, a.resize((160, 120)), "same.jpg")
    assert exif_thumbnail(same.read_bytes()) is not None
    assert not ThumbnailCheckProbe().run(same).detected
    other = _jpeg_with_thumbnail(tmp_path, a, b.resize((160, 120)), "other.jpg")
    f = ThumbnailCheckProbe().run(other)
    assert f.detected and "looks different" in f.summary
    cropped = _jpeg_with_thumbnail(tmp_path, a.crop((0, 0, 200, 300)), a.resize((160, 120)), "crop.jpg")
    assert ThumbnailCheckProbe().run(cropped).detected


def test_screen_sized_png_without_camera_tags_is_a_screenshot_and_a_photo_is_not(tmp_path):
    shot = tmp_path / "s.png"
    Image.fromarray((np.random.default_rng(0).random((800, 1280, 3)) * 40).astype(np.uint8)).save(shot)
    f = ScreenCaptureProbe().run(shot)
    assert f.detected and "screenshot" in f.summary
    photo = tmp_path / "p.jpg"
    Image.fromarray((np.random.default_rng(1).random((600, 900, 3)) * 255).astype(np.uint8)).save(photo)
    assert not ScreenCaptureProbe().run(photo).detected


def test_moire_peaks_score_a_grid_beat_higher_than_a_plain_picture():
    y, x = np.mgrid[0:600, 0:600]
    grid = 128 + 100 * np.sin(2 * np.pi * x / 7.3) * np.sin(2 * np.pi * y / 7.1)
    plain = np.random.default_rng(0).normal(128, 30, (600, 600))
    assert moire_peaks(Image.fromarray(np.clip(grid, 0, 255).astype(np.uint8))) > moire_peaks(
        Image.fromarray(np.clip(plain, 0, 255).astype(np.uint8)))


def test_burned_in_dates_coordinates_and_watermark_names_are_read_from_text():
    words = [(w, 90) for w in "REC 2024-06-01 12:34:56 N 40.71280 W 74.00600 GoPro Shutterstock".split()]
    c = clues(words)
    assert c["timestamps"] == ["2024-06-01 12:34:56"] and "GoPro" in c["marks"] and "Shutterstock" in c["marks"]
    assert c["coords"] and abs(c["coords"][0][0] - 40.7128) < 1e-6 and abs(c["coords"][0][1] + 74.006) < 1e-6
    assert parse_coords("48°51'24\" N 2°21'03\" E")[0][0] > 48.8
    assert parse_coords("price 12.3456 and 9") == [] and DATE_RX.search("version 2024-13-45") is None
