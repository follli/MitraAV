import io
import shutil
import struct
import subprocess
import zipfile
import zlib

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from mitraav.core import stego_container as sc
from mitraav.core.stego_stats import chi_square_prefix, prefix_verdict, rs_estimate
from mitraav.probes.stego_audio import StegoAudioProbe
from mitraav.probes.stego_container import StegoContainerProbe
from mitraav.probes.stego_image import StegoImageProbe
from mitraav.probes.stego_video import StegoVideoProbe, sei_user_data

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
ZIP = io.BytesIO()
with zipfile.ZipFile(ZIP, "w") as _z:
    _z.writestr("secret.txt", "meet at six " * 40)
ZIP = ZIP.getvalue()


def natural(h=300, w=400, seed=0, noise=1.6):
    """A photo-like picture: smooth colour fields plus a little sensor noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + 60 * np.sin(x / 37.0 + y / 91.0) + 40 * np.cos(y / 23.0) + 0.1 * x
    img = np.stack([base + 10 * c + 8 * np.sin(x / (15 + 7 * c)) for c in range(3)], axis=-1)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def lsb_embed(arr, rate, sequential=True, seed=1):
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1).copy()
    n = int(len(flat) * rate)
    idx = np.arange(n) if sequential else np.sort(rng.choice(len(flat), n, replace=False))
    flat[idx] = (flat[idx] & 0xFE) | rng.integers(0, 2, n).astype(np.uint8)
    return flat.reshape(arr.shape)


# --- container ----------------------------------------------------------------------------------------
def _jpeg(tmp_path, name="a.jpg"):
    p = tmp_path / name
    Image.fromarray(natural()).save(p, quality=90)
    return p


def test_data_appended_to_jpeg_png_gif_is_found_and_a_clean_file_is_not(tmp_path):
    for fmt, ext in (("JPEG", "jpg"), ("PNG", "png"), ("GIF", "gif")):
        clean = tmp_path / f"clean.{ext}"
        Image.fromarray(natural()).save(clean, format=fmt)
        assert StegoContainerProbe().run(clean).detected is False
        bad = tmp_path / f"zip.{ext}"
        bad.write_bytes(clean.read_bytes() + ZIP)
        f = StegoContainerProbe().run(bad)
        assert f.detected and "follow the end" in f.summary and "ZIP archive" in f.summary, fmt


def test_a_few_padding_bytes_are_ignored(tmp_path):
    p = _jpeg(tmp_path)
    p.write_bytes(p.read_bytes() + b"\n\n")
    assert not StegoContainerProbe().run(p).detected


def test_png_private_chunk_big_text_and_bad_crc(tmp_path):
    clean = tmp_path / "c.png"
    Image.fromarray(natural()).save(clean)
    d = clean.read_bytes()
    iend = d.rfind(b"IEND") - 4
    body = bytes(np.random.default_rng(0).integers(0, 256, 3000, dtype=np.uint8))
    chunk = struct.pack(">I", len(body)) + b"stEg" + body + struct.pack(">I", zlib.crc32(b"stEg" + body))
    (tmp_path / "chunk.png").write_bytes(d[:iend] + chunk + d[iend:])
    kinds = {o.kind for o in sc.inspect_file(tmp_path / "chunk.png")}
    assert "unknown_chunk" in kinds
    bad = d[:iend] + struct.pack(">I", 3) + b"tEXt" + b"abc" + b"\x00\x00\x00\x00" + d[iend:]
    (tmp_path / "crc.png").write_bytes(bad)
    assert "bad_crc" in {o.kind for o in sc.inspect_file(tmp_path / "crc.png")}


def test_jpeg_comment_with_a_payload_and_tool_names(tmp_path):
    p = _jpeg(tmp_path)
    j = p.read_bytes()
    com = b"\xff\xfe" + struct.pack(">H", 2 + 2500) + bytes(np.random.default_rng(0).integers(0, 256, 2500, dtype=np.uint8))
    (tmp_path / "c.jpg").write_bytes(j[:2] + com + j[2:])
    assert "comment" in {o.kind for o in sc.inspect_file(tmp_path / "c.jpg")}
    named = b"\xff\xfe" + struct.pack(">H", 2 + 20) + b"made with steghide"
    (tmp_path / "n.jpg").write_bytes(j[:2] + named + j[2:])
    assert "tool_name" in {o.kind for o in sc.inspect_file(tmp_path / "n.jpg")}


def test_a_phone_style_multi_picture_jpeg_is_not_flagged(tmp_path):
    p = _jpeg(tmp_path)
    j = p.read_bytes()
    mpf = b"\xff\xe2" + struct.pack(">H", 2 + 8) + b"MPF\x00MM\x00*"
    (tmp_path / "mpo.jpg").write_bytes(j[:2] + mpf + j[2:] + j)            # a second full JPEG after the first
    assert sc.inspect_file(tmp_path / "mpo.jpg") == []
    (tmp_path / "plain.jpg").write_bytes(j + j)                          # the same without MPF: appended image
    assert "trailing_data" in {o.kind for o in sc.inspect_file(tmp_path / "plain.jpg")}


def test_wav_chunks_filler_and_appended_data(tmp_path):
    p = tmp_path / "a.wav"
    wavfile.write(p, 8000, (np.random.default_rng(0).normal(0, 800, 8000)).astype(np.int16))
    d = p.read_bytes()
    assert sc.inspect_file(p) == []
    junk = b"JUNK" + struct.pack("<I", 2048) + bytes(np.random.default_rng(1).integers(1, 255, 2048, dtype=np.uint8))
    inserted = d[:12] + junk + d[12:]
    inserted = inserted[:4] + struct.pack("<I", len(inserted) - 8) + inserted[8:]
    (tmp_path / "junk.wav").write_bytes(inserted)
    assert "filler_not_empty" in {o.kind for o in sc.inspect_file(tmp_path / "junk.wav")}
    (tmp_path / "tail.wav").write_bytes(d + ZIP)
    assert "trailing_data" in {o.kind for o in sc.inspect_file(tmp_path / "tail.wav")}


def _mp4_with(boxes: bytes) -> bytes:
    ftyp = struct.pack(">I4s", 24, b"ftyp") + b"isom\x00\x00\x02\x00isomiso2"
    mdat = struct.pack(">I4s", 8 + 16, b"mdat") + b"\x00" * 16
    return ftyp + boxes + mdat


def test_mp4_free_space_that_is_not_empty_and_appended_bytes(tmp_path):
    empty = struct.pack(">I4s", 8 + 4096, b"free") + b"\x00" * 4096
    full = struct.pack(">I4s", 8 + 4096, b"free") + bytes(np.random.default_rng(0).integers(1, 255, 4096, dtype=np.uint8))
    (tmp_path / "e.mp4").write_bytes(_mp4_with(empty))
    (tmp_path / "f.mp4").write_bytes(_mp4_with(full))
    (tmp_path / "t.mp4").write_bytes(_mp4_with(empty) + ZIP)
    assert sc.inspect_file(tmp_path / "e.mp4") == []
    assert "filler_not_empty" in {o.kind for o in sc.inspect_file(tmp_path / "f.mp4")}
    assert "trailing_data" in {o.kind for o in sc.inspect_file(tmp_path / "t.mp4")}


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_matroska_attachment_is_reported(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("hidden")
    out = tmp_path / "a.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1",
                    "-c:v", "libx264", "-attach", str(secret), "-metadata:s:t", "mimetype=text/plain", str(out)], check=True)
    f = StegoContainerProbe().run(out)
    assert f.detected and "attached file" in f.summary


# --- pixels -------------------------------------------------------------------------------------------
def test_sequential_lsb_replacement_is_found_by_chi_square_and_rs():
    cover = natural()
    pre = chi_square_prefix(cover.reshape(-1), steps=20)
    assert prefix_verdict(pre) is None
    stego = lsb_embed(cover, 0.4)
    verdict = prefix_verdict(chi_square_prefix(stego.reshape(-1), steps=20))
    assert verdict and verdict[0] == "partial" and 0.3 <= verdict[1] <= 0.55
    est = [rs_estimate(stego[..., c]) for c in range(3)]
    assert all(e is not None and 0.2 <= e <= 0.6 for e in est)        # about 40% of the LSBs replaced
    assert all(abs(rs_estimate(cover[..., c])) < 0.06 for c in range(3))


def test_pixel_probe_flags_lsb_stego_in_png_and_bmp_and_passes_clean_ones(tmp_path):
    for fmt in ("PNG", "BMP"):
        clean = tmp_path / f"clean.{fmt.lower()}"
        Image.fromarray(natural()).save(clean)
        assert not StegoImageProbe().run(clean).detected, fmt
        for seq in (True, False):
            p = tmp_path / f"s{int(seq)}.{fmt.lower()}"
            Image.fromarray(lsb_embed(natural(), 0.5, sequential=seq)).save(p)
            f = StegoImageProbe().run(p)
            assert f.detected and "lowest bits" in f.summary and f.follow_ups, (fmt, seq)


def test_a_pure_noise_picture_is_not_called_hidden_data(tmp_path):
    p = tmp_path / "noise.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)).save(p)
    assert not StegoImageProbe().run(p).detected


def test_alpha_channel_lsb_bits_are_flagged(tmp_path):
    rgb = natural()
    alpha = np.where(np.random.default_rng(0).random(rgb.shape[:2]) < 0.5, 254, 255).astype(np.uint8)
    p = tmp_path / "a.png"
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(p)
    f = StegoImageProbe().run(p)
    assert f.detected and "transparency channel" in f.summary


def test_jpeg_pixels_are_skipped_without_the_addon_and_never_crash(tmp_path, monkeypatch):
    import mitraav.core.stego_dct as dct
    monkeypatch.setattr(dct, "dct_available", lambda: False)
    f = StegoImageProbe().run(_jpeg(tmp_path))
    assert not f.detected and f.info and "lossy" in f.summary and "stego add-on" in f.summary


jpeglib = pytest.importorskip("jpeglib", reason="the stego add-on (jpeglib) is optional") if False else None


def _has_jpeglib():
    import importlib.util
    return importlib.util.find_spec("jpeglib") is not None


@pytest.mark.skipif(not _has_jpeglib(), reason="the stego add-on (jpeglib) is optional")
def test_dct_tests_find_jsteg_and_f5_style_changes_and_pass_a_clean_jpeg(tmp_path):
    import jpeglib
    src = tmp_path / "busy.jpg"
    Image.fromarray(natural(600, 800, noise=9)).save(src, quality=90)             # enough coefficients to test
    assert not StegoImageProbe().run(src).detected
    im = jpeglib.read_dct(str(src))
    y = im.Y.copy().reshape(-1, 64)
    rng = np.random.default_rng(0)
    ok = np.argwhere((y[:, 1:] != 0) & (y[:, 1:] != 1))
    for b, k in ok[: int(len(ok) * 0.5)]:
        y[b, k + 1] = (y[b, k + 1] & ~1) | int(rng.integers(0, 2))
    im.Y = y.reshape(im.Y.shape)
    im.write_dct(str(tmp_path / "jsteg.jpg"))
    f = StegoImageProbe().run(tmp_path / "jsteg.jpg")
    assert f.detected and "DCT coefficients" in f.summary
    im2 = jpeglib.read_dct(str(src))
    y2 = im2.Y.copy().reshape(-1, 64)
    nz = np.argwhere(y2[:, 1:] != 0)
    for b, k in nz[rng.random(len(nz)) < 0.4]:
        y2[b, k + 1] -= np.sign(y2[b, k + 1])
    im2.Y = y2.reshape(im2.Y.shape)
    im2.write_dct(str(tmp_path / "f5.jpg"))
    f5 = StegoImageProbe().run(tmp_path / "f5.jpg")
    assert f5.detected and "shrinkage" in f5.summary


# --- audio --------------------------------------------------------------------------------------------
def _audio(sr=44100, secs=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(sr * secs) / sr
    x = 20 * np.sin(2 * np.pi * 220 * t) + rng.laplace(0, 30, t.size)        # peaked like speech and room noise
    return np.clip(x, -32000, 32000).astype(np.int16), sr


def test_lsb_replacement_in_a_16_bit_wav_is_found(tmp_path):
    x, sr = _audio()
    clean = tmp_path / "clean.wav"
    wavfile.write(clean, sr, x)
    assert not StegoAudioProbe().run(clean).detected
    y = x.copy()
    n = int(len(y) * 0.3)
    y[:n] = (y[:n] & ~1) | np.random.default_rng(1).integers(0, 2, n).astype(np.int16)
    p = tmp_path / "stego.wav"
    wavfile.write(p, sr, y)
    f = StegoAudioProbe().run(p)
    assert f.detected and "random message bits" in f.summary


def test_random_bits_in_digital_silence_and_a_hidden_tone_are_found(tmp_path):
    x, sr = _audio()
    x[sr * 2:sr * 4] = 0                                           # two seconds of digital silence
    clean = tmp_path / "gap.wav"
    wavfile.write(clean, sr, x)
    assert not StegoAudioProbe().run(clean).detected
    y = x.copy()
    y[sr * 2:sr * 4] = np.random.default_rng(2).integers(0, 2, sr * 2).astype(np.int16)
    wavfile.write(tmp_path / "gapbits.wav", sr, y)
    assert "near-silence" in StegoAudioProbe().run(tmp_path / "gapbits.wav").summary
    t = np.arange(len(x)) / sr
    z = (x + 600 * np.sin(2 * np.pi * 19000 * t)).astype(np.int16)
    wavfile.write(tmp_path / "tone.wav", sr, z)
    f = StegoAudioProbe().run(tmp_path / "tone.wav")
    assert f.detected and "19.0 kHz" in f.summary and f.regions


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_lossy_audio_skips_the_lsb_tests_with_a_reason(tmp_path):
    x, sr = _audio()
    wavfile.write(tmp_path / "a.wav", sr, x)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "a.wav"), str(tmp_path / "a.ogg")], check=True)
    f = StegoAudioProbe().run(tmp_path / "a.ogg")
    assert not f.detected and "LSB tests skipped" in f.summary


# --- video --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_unfamiliar_sei_user_data_in_h264_is_reported(tmp_path):
    base = tmp_path / "base.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base)], check=True)
    assert not StegoVideoProbe().run(base).detected
    payload = "x" * 400
    out = tmp_path / "sei.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(base), "-c", "copy", "-bsf:v",
                    f"h264_metadata=sei_user_data=12345678-1234-1234-1234-123456789abc+{payload}", str(out)], check=True)
    f = StegoVideoProbe().run(out)
    assert f.detected and "12345678123412341234123456789abc" in f.summary


def test_sei_parser_reads_user_data_unregistered_messages():
    uuid = bytes(range(16))
    payload = uuid + b"hello hidden world"
    sei = b"\x00\x00\x01\x06" + bytes([5, len(payload)]) + payload + b"\x80"
    got = sei_user_data(b"\x00\x00\x00\x01\x67\x42" + sei + b"\x00\x00\x01\x65\x88", hevc=False)
    assert got == {uuid: [len(b"hello hidden world")]}


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_lsb_replacement_in_lossless_video_frames_is_found(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    for i in range(4):
        Image.fromarray(lsb_embed(natural(300, 400, seed=i), 0.6)).save(frames / f"{i:03d}.png")
    clean_dir = tmp_path / "c"
    clean_dir.mkdir()
    for i in range(4):
        Image.fromarray(natural(300, 400, seed=i)).save(clean_dir / f"{i:03d}.png")
    for name, d in (("stego", frames), ("clean", clean_dir)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "5", "-i", str(d / "%03d.png"), "-c:v", "ffv1",
                        "-pix_fmt", "bgr0", str(tmp_path / f"{name}.mkv")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "5", "-i", str(clean_dir / "%03d.png"), "-c:v", "ffv1",
                    "-pix_fmt", "yuv444p", str(tmp_path / "clean_yuv.mkv")], check=True)
    assert not StegoVideoProbe().run(tmp_path / "clean.mkv").detected
    assert not StegoVideoProbe().run(tmp_path / "clean_yuv.mkv").detected          # luma plane read natively
    f = StegoVideoProbe().run(tmp_path / "stego.mkv")
    assert f.detected and "sampled frames" in f.summary


def test_findings_are_registered_as_probes_in_their_own_group():
    from mitraav.core.runner import discover_probes
    names = {c.name: c for c in discover_probes()}
    for n in ("hidden_data_container", "hidden_data_pixels", "hidden_data_audio", "hidden_data_video"):
        assert names[n].group == "Hidden data"


def test_large_files_are_memory_mapped_with_the_same_result(tmp_path, monkeypatch):
    import struct

    from mitraav.core import stego_container as sc
    pcm = b"\x01\x00" * 40000
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    wav = b"RIFF" + struct.pack("<I", len(body)) + body + b"PK\x03\x04" + bytes(range(256)) * 8
    p = tmp_path / "t.wav"
    p.write_bytes(wav)
    whole = [(o.kind, o.offset, o.size) for o in sc.inspect_file(p)]
    monkeypatch.setattr(sc, "LOAD_LIMIT", 1000)
    mapped = [(o.kind, o.offset, o.size) for o in sc.inspect_file(p)]
    assert whole == mapped and any(k == "trailing_data" for k, _, _ in whole)
