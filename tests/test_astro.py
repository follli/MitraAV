from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from mitraav.core.astro import compass, moon_phase, sun_position
from mitraav.probes.sun_moon import SunMoonProbe, image_time_place


def test_sun_position_matches_known_geometry():
    e, _ = sun_position(0, 0, datetime(2024, 3, 20, 12, 0, tzinfo=UTC))
    assert abs(e - 88.2) < 1.5                                   # near the equinox the sun is overhead on the equator
    e, az = sun_position(51.5, -0.12, datetime(2024, 6, 21, 12, 0, tzinfo=UTC))
    assert abs(e - 61.9) < 0.6 and abs(az - 179) < 3             # London solar noon at the solstice
    assert 80 < sun_position(51.5, -0.12, datetime(2024, 6, 21, 7, 0, tzinfo=UTC))[1] < 90      # morning: east
    assert 270 < sun_position(51.5, -0.12, datetime(2024, 6, 21, 17, 0, tzinfo=UTC))[1] < 278  # evening: west
    assert compass(179) == "south" and compass(271) == "west"


def test_moon_phase_at_new_and_full_moon():
    assert moon_phase(datetime(2024, 4, 8, 18, 21, tzinfo=UTC))[0] < 0.03
    assert moon_phase(datetime(2024, 4, 23, 23, 49, tzinfo=UTC))[0] > 0.97


def _jpeg_with_exif(tmp_path, when, lat, lon, brightness, name="p.jpg", offset=None, gps_time=None):
    img = Image.fromarray(np.full((60, 80, 3), brightness, np.uint8))
    ex = Image.Exif()
    sub = ex.get_ifd(0x8769)
    sub[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    if offset:
        sub[0x9011] = offset
    gps = ex.get_ifd(0x8825)
    gps[1], gps[3] = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    la, lo = abs(lat), abs(lon)
    gps[2] = (int(la), int((la % 1) * 60), ((la % 1) * 60 % 1) * 60)
    gps[4] = (int(lo), int((lo % 1) * 60), ((lo % 1) * 60 % 1) * 60)
    if gps_time:
        gps[29], gps[7] = gps_time.strftime("%Y:%m:%d"), (gps_time.hour, gps_time.minute, gps_time.second)
    p = tmp_path / name
    img.save(p, exif=ex)
    return p


def test_time_zone_comes_from_gps_utc_then_the_offset_then_the_longitude(tmp_path):
    t = datetime(2024, 6, 1, 12, 0)
    w, *_ = image_time_place(_jpeg_with_exif(tmp_path, t, 51.5, -0.12, 100, gps_time=datetime(2024, 6, 1, 11, 0)))
    assert w.utcoffset() == timedelta(0) and w.hour == 11
    w, *_, how = image_time_place(_jpeg_with_exif(tmp_path, t, 51.5, -0.12, 100, "o.jpg", offset="+01:00"))
    assert w.utcoffset() == timedelta(hours=1) and "own UTC offset" in how
    w, *_, how = image_time_place(_jpeg_with_exif(tmp_path, t, 51.5, 15.0, 100, "g.jpg"))
    assert w.utcoffset() == timedelta(hours=1) and "guess" in how


def test_probe_gives_sun_numbers_and_only_warns_about_blatant_mismatches(tmp_path):
    day = _jpeg_with_exif(tmp_path, datetime(2024, 6, 21, 13, 0), 51.5, -0.12, 120, offset="+01:00")
    f = SunMoonProbe().run(day)
    assert not f.detected and f.info and "above the horizon" in f.summary and "shadow" in f.summary and f.follow_ups
    night_bright = _jpeg_with_exif(tmp_path, datetime(2024, 6, 21, 1, 30), 51.5, -0.12, 200, "n.jpg", offset="+01:00")
    g = SunMoonProbe().run(night_bright)
    assert g.detected and "does not fit" in g.summary
    none = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(none)
    assert not SunMoonProbe().run(none).detected and Path(none).exists()
