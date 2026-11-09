import shutil
import struct
import subprocess

import numpy as np
import pytest

from mitraav.probes.camera_motion import CameraMotionProbe, classify, global_shifts
from mitraav.probes.video_cuts import VideoCutsProbe
from mitraav.probes.video_telemetry import (
    VideoTelemetryProbe,
    parse_gpmf,
    parse_nmea,
    summarise_track,
)
from mitraav.probes.video_text_tracks import VideoTextTracksProbe, guess_language, parse_srt

HAVE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE, reason="needs ffmpeg")


def klv(key: bytes, typ: str, size: int, repeat: int, data: bytes) -> bytes:
    body = struct.pack(">4sBBH", key, ord(typ) if typ != "\0" else 0, size, repeat) + data
    return body + b"\x00" * (-len(data) % 4)


def gpmf_block(points):
    scal = klv(b"SCAL", "l", 4, 5, struct.pack(">5i", 10_000_000, 10_000_000, 1000, 1000, 100))
    rows = b"".join(struct.pack(">5i", int(la * 1e7), int(lo * 1e7), int(al * 1000), int(sp * 1000), int(sp * 100))
                    for la, lo, al, sp in points)
    gps = klv(b"GPS5", "l", 20, len(points), rows)
    strm = scal + gps
    inner = klv(b"STRM", "\0", 1, len(strm), strm)
    return klv(b"DEVC", "\0", 1, len(inner), inner)


def test_gpmf_gps5_samples_are_decoded_and_summarised():
    pts = [(40.7128, -74.0060, 10.0, 12.0), (40.7138, -74.0050, 11.0, 13.0), (40.7148, -74.0040, 12.0, 14.0)]
    got = parse_gpmf(gpmf_block(pts))
    assert len(got) == 3 and abs(got[0]["lat"] - 40.7128) < 1e-6 and abs(got[2]["lon"] + 74.004) < 1e-6
    text, first = summarise_track(got, "GoPro GPS track")
    assert "3 position samples" in text and first["lat"] == got[0]["lat"]


def test_nmea_rmc_sentences_give_position_speed_and_time():
    s = b"junk$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\nmore"
    pts = parse_nmea(s)
    assert len(pts) == 1 and abs(pts[0]["lat"] - 48.1173) < 1e-3 and abs(pts[0]["lon"] - 11.5167) < 1e-3
    assert "1994-03-23 12:35:19" in pts[0]["time"] and abs(pts[0]["speed"] - 11.52) < 0.05
    assert parse_nmea(b"$GPRMC,123519,V,4807.038,N,01131.000,E,0,0,230394,,*00") == []


@needs_ffmpeg
def test_nmea_inside_a_video_file_is_found_and_hinted(tmp_path):
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "libx264", str(v)], check=True)
    v.write_bytes(v.read_bytes() + b"\x00\x00\x00\x08free$GPRMC,123519,A,5130.000,N,00007.000,W,010.0,084.4,010624,003.1,W*00")
    f = VideoTelemetryProbe().run(v)
    assert f.detected and "NMEA" in f.summary and f.meta["geo"]["any_of"] == ["europe"]
    assert any("openstreetmap" in u.url for u in f.follow_ups)
    clean = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "libx264", str(clean)], check=True)
    assert not VideoTelemetryProbe().run(clean).detected


def test_srt_parsing_and_language_guess():
    srt = "1\n00:00:01,000 --> 00:00:03,000\nThe cat is on the table and the dog is in the garden.\n\n2\n00:00:05,000 --> 00:00:07,500\nThis is what they said about the weather and the road.\n"
    words, t0, t1, n = parse_srt(srt)
    assert n == 2 and t0 == 1 and t1 == 7 and "weather" in words
    assert guess_language(words) == "English"


@needs_ffmpeg
def test_subtitle_track_is_read_without_printing_the_text(tmp_path):
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nThe cat is on the table and the dog is in the garden.\n\n"
                   "2\n00:00:02,500 --> 00:00:04,000\nThis is what they said about the weather on 2024-06-01 and the road.\n")
    v = tmp_path / "v.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=5", "-i", str(srt),
                    "-c:v", "libx264", "-c:s", "srt", str(v)], check=True)
    f = VideoTextTracksProbe().run(v)
    assert f.detected and "2 cues" in f.summary and "English" in f.summary and "2024-06-01" in f.summary
    assert "cat is on the table" not in f.summary
    plain = tmp_path / "p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "libx264", str(plain)], check=True)
    assert not VideoTextTracksProbe().run(plain).detected


@needs_ffmpeg
def test_hard_cuts_are_counted_and_a_single_shot_is_not_a_montage(tmp_path):
    srcs = ["testsrc=size=128x96:rate=10:duration=2", "color=c=white:size=128x96:rate=10:duration=2",
            "smptebars=size=128x96:rate=10:duration=2", "color=c=black:size=128x96:rate=10:duration=2"]
    parts = []
    for i, s in enumerate(srcs):
        p = tmp_path / f"p{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", s, "-t", "2", "-c:v", "libx264", str(p)], check=True)
        parts.append(p)
    lst = tmp_path / "l.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = tmp_path / "cat.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    f = VideoCutsProbe().run(out)
    assert f.detected and f.regions and "hard cuts" in f.summary
    assert not VideoCutsProbe().run(parts[0]).detected


def _motion_video(tmp_path, name, expr_x, expr_y):
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "mandelbrot=size=640x480:rate=10", "-t", "6",
                    "-vf", f"scale=1280:960,crop=640:480:x='{expr_x}':y='{expr_y}'", "-c:v", "libx264", str(out)], check=True)
    return out


@needs_ffmpeg
def test_camera_motion_separates_fixed_pan_and_shake(tmp_path):
    fixed = _motion_video(tmp_path, "fixed.mp4", "320", "240")
    pan = _motion_video(tmp_path, "pan.mp4", "50+60*t", "240")
    shake = _motion_video(tmp_path, "shake.mp4", "320+8*sin(40*t)+6*sin(23*t)", "240+7*sin(31*t)")
    assert "fixed camera" in CameraMotionProbe().run(fixed).summary
    assert "pan" in CameraMotionProbe().run(pan).summary
    assert "hand-held" in CameraMotionProbe().run(shake).summary


def test_classify_handles_short_series():
    assert classify(np.zeros((5, 2)))[0] == "unknown"
    assert global_shifts(np.random.default_rng(0).random((4, 32, 32))).shape == (3, 2)
