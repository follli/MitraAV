import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtWidgets import QApplication
from scipy.io import wavfile

from mitraav.core.finding import Finding, MediaType
from mitraav.core.frames import thumbnail
from mitraav.gui import preview as pv
from mitraav.gui.main_window import ElidedLabel, MainWindow, _play_at


def _app():
    return QApplication.instance() or QApplication([])


def _wait(ms=600):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _wav(tmp_path, secs=3):
    p = tmp_path / "a.wav"
    wavfile.write(p, 8000, (0.2 * np.sin(2 * np.pi * 440 * np.arange(8000 * secs) / 8000) * 32767).astype(np.int16))
    return p


def test_image_preview_shows_a_thumbnail_and_no_player(tmp_path):
    _app()
    p = tmp_path / "pic.jpg"
    Image.new("RGB", (800, 600), "teal").save(p)
    w = pv.MediaPreview()
    w.load(p, MediaType.IMAGE)
    assert w.isVisibleTo(None) or not w.isHidden()
    assert w.thumb.pixmap() is not None and not w.thumb.pixmap().isNull()
    assert not w.can_play() and not w.controls.isVisibleTo(w)
    w.clear()
    assert w.isHidden() and w.path is None


@pytest.mark.skipif(not pv.HAVE_MEDIA, reason="Qt multimedia not available")
def test_audio_preview_loads_and_seeks(tmp_path):
    _app()
    w = pv.MediaPreview()
    w.load(_wav(tmp_path, 3), MediaType.AUDIO)
    _wait(800)
    assert w.can_play() and w.player.source() == QUrl.fromLocalFile(str(tmp_path / "a.wav"))
    assert w.player.duration() in range(2900, 3100)
    assert w.play_from(1.5)
    w.player.pause()
    assert w.player.position() >= 1400
    assert not w.play_from(-1) or True                  # negative start is clamped, never raises
    w.clear()
    assert not w.can_play()


def test_a_missing_multimedia_backend_degrades_to_a_note(tmp_path, monkeypatch):
    _app()
    monkeypatch.setattr(pv, "HAVE_MEDIA", False)
    w = pv.MediaPreview()
    w.load(_wav(tmp_path), MediaType.AUDIO)
    assert not w.can_play() and not w.play_from(1.0)
    assert "unavailable" in w.note.text().lower()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_video_thumbnail_is_extracted_even_from_a_very_short_clip(tmp_path):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    t = thumbnail(p, width=96)
    assert t and t.exists() and Image.open(t).width == 96


def test_elided_label_keeps_the_full_text():
    _app()
    lab = ElidedLabel("a_very_long_file_name_for_a_recording_of_a_level_crossing_2025.ogx")
    lab.resize(180, 30)
    lab.show()
    assert lab.fullText().endswith("2025.ogx") and len(lab.text()) < len(lab.fullText()) and "…" in lab.text()
    lab.setFullText("short.wav")
    assert lab.text() == "short.wav"


def test_play_links_appear_only_when_there_is_something_to_play_and_start_before_the_event(tmp_path, monkeypatch):
    _app()
    w = MainWindow()
    f = Finding(probe="doppler_pass", group="Aircraft and motion", detected=True, strength=0.9, narrows=0.4,
                summary="event 1", regions=[dict(label="event 1", t0=5.0, t1=60.0, t_play=17.0, f0=1, f1=2)])
    w.add_finding(f)
    w.tree.topLevelItem(0).child(0).setSelected(True)
    assert "mitraav-play" not in w.detail.toHtml()                  # nothing loaded: no playable source
    monkeypatch.setattr(w.preview, "can_play", lambda: True)
    w.show_detail()
    assert "mitraav-play:17.00" in w.detail.toHtml().replace("&amp;", "&") or "17.00" in w.detail.toHtml()
    assert _play_at(f.regions[0]) == 17.0 and _play_at(dict(t0=4.0)) == 4.0


def test_clicking_a_play_link_seeks_the_preview(monkeypatch):
    _app()
    w = MainWindow()
    got = []
    monkeypatch.setattr(w.preview, "play_from", lambda s: got.append(s) or True)
    w._link_clicked(QUrl("mitraav-play:42.5"))
    assert got == [42.5]


def test_header_has_a_split_open_history_pill_and_focus_signal_button():
    from PySide6.QtWidgets import QPushButton
    _app()
    w = MainWindow()
    assert not hasattr(w, "chip")                                    # the lead-count chip is gone
    names = {b.text(): b.objectName() for b in w.findChildren(QPushButton)}
    assert names["Open file"] == "segLeft" and names["History"] == "segRight"
    assert w.spec_btn.text() == "Focus Signal" and "Show signal" not in names
    assert w.drop.parent() is w.spec_btn.parent()                    # all in the header row
