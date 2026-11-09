import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from mitraav.probes.image_tamper import ImageTamperProbe, copy_move, ela_outliers


def tex(seed, sig=2.0, amp=55, shape=(600, 800)):
    b = gaussian_filter(np.random.default_rng(seed).normal(0, 1, shape), sig)
    return np.clip(128 + amp * b / b.std(), 0, 255).astype(np.uint8)


def jpg(a, q):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("L")


def test_a_cloned_patch_is_found_even_after_jpeg_and_a_clean_picture_is_not():
    img = tex(0)
    assert copy_move(img.astype(float)) == []
    cloned = img.copy()
    cloned[400:500, 500:620] = img[100:200, 100:220]
    hits = copy_move(np.asarray(jpg(cloned, 90), float))
    assert hits and (hits[0]["dx"], hits[0]["dy"]) == (400, 300) and hits[0]["area"] > 8000


def test_periodic_patterns_are_not_called_clones():
    y, x = np.mgrid[0:600, 0:800]
    tiles = (128 + 60 * np.sin(x / 11.0) * np.sin(y / 13.0) + np.random.default_rng(0).normal(0, 6, (600, 800)))
    assert copy_move(tiles) == []


def test_error_level_outlier_marks_a_patch_with_another_compression_history():
    base = np.asarray(jpg(tex(1), 95))
    donor = np.asarray(jpg(tex(2), 40))
    comp = base.copy()
    comp[200:380, 300:560] = donor[200:380, 300:560]
    assert ela_outliers(jpg(base, 95)) == []
    out = ela_outliers(jpg(comp, 95))
    assert out and out[0]["bbox"][0] < 330 and out[0]["bbox"][2] > 520


def test_probe_reports_boxes_for_both_clone_and_source_and_ignores_screenshots(tmp_path):
    img = tex(0)
    img[400:500, 500:620] = img[100:200, 100:220]
    p = tmp_path / "c.png"
    Image.fromarray(img).save(p)
    f = ImageTamperProbe().run(p)
    assert f.detected and len(f.boxes) == 2 and "reappears shifted" in f.summary
    ui = np.full((600, 800), 255, np.uint8)
    for y in range(20, 580, 60):
        ui[y:y + 12, 40:300] = 30
    q = tmp_path / "ui.png"
    Image.fromarray(ui).save(q)
    assert not ImageTamperProbe().run(q).detected
