import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from mitraav.core.audio import iter_chunks
from mitraav.core.finding import MediaType, plain
from mitraav.core.lines import find_lines
from mitraav.core.runner import media_of
from mitraav.probes import nature_id as ni
from mitraav.probes.image_text import clues
from mitraav.probes.signal_tones import SignalTonesProbe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@needs_ffmpeg
def test_ogg_with_an_ogx_name_is_sniffed_as_audio(tmp_path):
    p = tmp_path / "bells.ogx"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                        "-c:a", "libvorbis", "-f", "ogg", str(p)], capture_output=True)
    if r.returncode:
        pytest.skip("this ffmpeg has no Vorbis encoder")
    assert media_of(p) == MediaType.AUDIO


@needs_ffmpeg
def test_ogg_video_is_sniffed_as_video(tmp_path):
    p = tmp_path / "crossing.ogx"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
                        "-c:v", "libtheora", "-f", "ogg", str(p)], capture_output=True)
    if r.returncode:
        pytest.skip("this ffmpeg has no Theora encoder")
    assert media_of(p) == MediaType.VIDEO


def test_known_and_unknown_extensions(tmp_path):
    assert media_of(Path("a.oga")) == MediaType.AUDIO and media_of(Path("a.ogv")) == MediaType.VIDEO
    assert media_of(Path("a.JPG")) == MediaType.IMAGE
    img = tmp_path / "photo.dat"                      # an image with a meaningless extension
    Image.new("RGB", (32, 32), "red").save(img, format="PNG")
    if shutil.which("ffprobe"):
        assert media_of(img) == MediaType.IMAGE
    assert media_of(tmp_path / "missing.xyz") is None


def _tone_file(tmp_path, secs, sr=8000, freqs=(350.0, 440.0)):
    t = np.arange(int(sr * secs)) / sr
    x = sum(0.25 * np.sin(2 * np.pi * f * t) for f in freqs)
    x = x + 0.002 * np.random.default_rng(0).standard_normal(t.size)
    p = tmp_path / "tone.wav"
    wavfile.write(p, sr, (x * 32767).astype(np.int16))
    return p


def test_a_short_clip_is_one_chunk_not_none(tmp_path):
    p = _tone_file(tmp_path, 3.0)
    chunks = list(iter_chunks(p))
    assert len(chunks) == 1 and abs(len(chunks[0][1]) / chunks[0][2] - 3.0) < 0.05


def test_a_five_second_strong_dial_tone_is_found(tmp_path):
    p = _tone_file(tmp_path, 5.0)
    lines = find_lines(*__import__("mitraav.core.audio", fromlist=["load_audio"]).load_audio(p, offset=0, duration=10), 0.0)
    got = sorted(round(ln.f) for ln in lines if ln.prom > 30)
    assert any(abs(g - 350) <= 4 for g in got) and any(abs(g - 440) <= 4 for g in got)   # thin lines, not blobs
    f = SignalTonesProbe().run(p)
    assert f.detected and "North American dial tone" in plain(f.summary)


def test_faint_codec_artefacts_do_not_block_a_tone_match(tmp_path):
    sr = 8000
    t = np.arange(sr * 5) / sr
    x = 0.25 * np.sin(2 * np.pi * 350 * t) + 0.25 * np.sin(2 * np.pi * 440 * t)
    for f in (496, 527, 560, 592):                       # artefacts about 40 dB down
        x += 0.0025 * np.sin(2 * np.pi * f * t)
    p = tmp_path / "t.wav"
    wavfile.write(p, sr, (x * 32767).astype(np.int16))
    assert SignalTonesProbe().run(p).detected


def test_ocr_noise_is_not_a_language():
    junk = ["A", "2", "at,", "&", "a", "Cat", "<2", "tis", "=", "3", "at", "=", "3", "%", "i", "4", "j", "way", "a", "NE", "¢", "x", "ey", "NG", "Te", "A", "=", "Z", "LAY"]
    c = clues([(w, 75) for w in junk])
    assert c["language"] is None and not c["real_text"] and c["words"] == []


def test_real_words_still_give_a_language():
    words = ["the", "market", "is", "open", "for", "the", "people", "and", "the", "street", "is", "open", "from", "the", "morning"]
    c = clues([(w, 90) for w in words])
    assert c["real_text"] and c["language"] == "English"


def test_species_guess_is_withheld_for_a_building_scene(tmp_path, monkeypatch):
    from mitraav.probes import scene_type as st
    names, _ = st.load_labels()
    p = np.zeros(365)
    p[names.index("tower")] = 0.6
    import importlib.util
    real = importlib.util.find_spec                    # the gate skips itself without a LiteRT runtime
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if name == "ai_edge_litert" else real(name, *a, **k))
    monkeypatch.setattr(st, "classify", lambda path, mp: p)
    monkeypatch.setattr("mitraav.core.models.places_path", lambda: Path(__file__))   # "exists"
    img = tmp_path / "cathedral.png"
    Image.new("RGB", (64, 64)).save(img)
    msg = ni._scene_gate(img)
    assert msg and "tower" in msg and "no species" in msg
    p[:] = 0
    p[names.index("forest/broadleaf")] = 0.8
    assert ni._scene_gate(img) is None                    # a woodland scene may show an organism


def _write(tmp_path, x, sr, name="x.wav"):
    p = tmp_path / name
    wavfile.write(p, sr, (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16))
    return p


def test_a_siren_with_harmonics_is_not_a_telephone_tone(tmp_path):
    sr = 16000
    t = np.arange(sr * 12) / sr
    bursts = ((t % 6) < 3).astype(float)
    x = sum(a * np.sin(2 * np.pi * k * 425 * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2))) * bursts
    x = x + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    f = SignalTonesProbe().run(_write(tmp_path, x, sr))
    assert "telephone" not in f.summary                  # a harmonic horn is measured, never matched to the table
    pure = np.sin(2 * np.pi * 425 * t) * bursts + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    assert SignalTonesProbe().run(_write(tmp_path, pure, sr, "pure.wav")).detected      # a pure tone still matches


def test_a_tonal_siren_is_not_a_jet_passby():
    from mitraav.probes.passby import analyse
    sr = 16000
    t = np.arange(sr * 100) / sr
    env = np.exp(-((t - 50) / 14) ** 2)                      # a smooth loudness bump, like a pass-by
    siren = env * np.sin(2 * np.pi * 600 * t) + 0.05 * np.random.default_rng(1).standard_normal(t.size)
    assert analyse(siren.astype(np.float32), sr) is None
    noise = env * np.random.default_rng(2).standard_normal(t.size) + 0.05 * np.random.default_rng(3).standard_normal(t.size)
    assert analyse(noise.astype(np.float32), sr) is not None  # broadband noise is still a pass-by


def test_a_siren_marks_pitch_and_pass_by_probes_inconclusive():
    from mitraav.core.runner import masking_caveat
    from mitraav.probes.doppler import DopplerProbe
    from mitraav.probes.hum import HumProbe
    from mitraav.probes.passby import BroadbandPassbyProbe
    from mitraav.probes.signal_tones import SignalTonesProbe as Tones
    shares = {"sirens": 0.62, "music": 0.05}
    for cls in (BroadbandPassbyProbe, DopplerProbe, Tones):
        note = masking_caveat(cls, shares)
        assert note and "siren" in note and "62%" in note
    assert masking_caveat(HumProbe, shares) is None                  # hum is not imitated by a siren
    assert masking_caveat(BroadbandPassbyProbe, {"sirens": 0.1}) is None
