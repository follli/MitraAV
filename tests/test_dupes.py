import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

from mitraav.cli import main
from mitraav.core.dupes import find_groups

HAVE = bool(shutil.which("ffmpeg"))


def photo(seed, shape=(300, 400)):
    r = np.random.default_rng(seed)
    base = gaussian_filter(r.normal(0, 1, shape), 6)
    return np.clip(128 + 60 * base / base.std(), 0, 255).astype(np.uint8)


def test_resized_recompressed_and_mirrored_copies_group_together_and_different_pictures_do_not(tmp_path):
    a = Image.fromarray(photo(1))
    a.save(tmp_path / "a.png")
    a.resize((200, 150)).save(tmp_path / "a_small.jpg", quality=60)
    a.transpose(Image.FLIP_LEFT_RIGHT).save(tmp_path / "a_mirror.png")
    Image.fromarray(photo(2)).save(tmp_path / "b.png")
    groups = find_groups(tmp_path)
    assert len(groups) == 1 and {p.name for p in groups[0]} == {"a.png", "a_small.jpg", "a_mirror.png"}


def test_a_quieter_copy_of_a_recording_groups_but_another_recording_does_not(tmp_path):
    sr = 16000
    t = np.arange(sr * 20) / sr
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * (300 + 200 * np.sin(2 * np.pi * 0.1 * t)) * t) * np.abs(np.sin(2 * np.pi * 0.2 * t)) + 0.05 * rng.standard_normal(len(t))
    y = np.sin(2 * np.pi * 900 * t) * np.abs(np.sin(2 * np.pi * 0.33 * t)) + 0.05 * rng.standard_normal(len(t))
    for name, sig, gain in (("x.wav", x, 1.0), ("x_quiet.wav", x, 0.3), ("y.wav", y, 1.0)):
        wavfile.write(tmp_path / name, sr, (sig * gain * 20000).astype(np.int16))
    groups = find_groups(tmp_path)
    assert len(groups) == 1 and {p.name for p in groups[0]} == {"x.wav", "x_quiet.wav"}


@pytest.mark.skipif(not HAVE, reason="needs ffmpeg")
def test_a_re_encoded_video_groups_with_the_original(tmp_path):
    src = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=4", "-c:v", "libx264", str(src)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=160:120", "-b:v", "50k", str(tmp_path / "a_small.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=10", "-t", "4", "-c:v", "libx264", str(tmp_path / "other.mp4")], check=True)
    groups = find_groups(tmp_path)
    assert len(groups) == 1 and {p.name for p in groups[0]} == {"a.mp4", "a_small.mp4"}


def test_cli_lists_groups_and_needs_a_folder(tmp_path, capsys):
    Image.fromarray(photo(1)).save(tmp_path / "a.png")
    Image.fromarray(photo(1)).save(tmp_path / "copy.png")
    assert main([str(tmp_path), "--duplicates"]) == 0
    out = capsys.readouterr().out
    assert "Group 1" in out and "a.png" in out and "copy.png" in out
