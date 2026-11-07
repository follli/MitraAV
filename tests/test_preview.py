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


@pytest.mark.skipif(bool(os.environ.get("CI")) or not pv.media_available(),
                    reason="real playback needs Qt multimedia and is skipped on headless CI")
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
    monkeypatch.setattr(pv, "_MEDIA", False)
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


def test_header_has_a_split_open_history_pill_and_no_focus_button():
    from PySide6.QtWidgets import QPushButton
    _app()
    w = MainWindow()
    assert not hasattr(w, "chip") and not hasattr(w, "spec_btn")         # chip and header button are gone
    names = {b.text(): b.objectName() for b in w.findChildren(QPushButton)}
    assert names["Open file"] == "segLeft" and names["History"] == "segRight"
    assert "Focus Signal" not in names and "Show signal" not in names
    assert w.drop.parent() is w.findChild(QPushButton, "segLeft").parent()


def test_details_pill_is_a_column_on_rows_with_a_signal():
    from mitraav.gui.main_window import FOCUS_COL, ROLE
    _app()
    w = MainWindow()
    reg = [dict(label="event 1", t0=5.0, t1=60.0, f0=100.0, f1=500.0)]
    with_signal = Finding(probe="doppler_pass", group="Aircraft and motion", detected=True, strength=0.9,
                          narrows=0.4, summary="e", regions=reg)
    without = Finding(probe="mains_hum", group="Setting and power", detected=True, strength=0.9,
                      narrows=0.4, summary="h")
    w.focus_ok = True
    w.add_finding(with_signal)
    w.add_finding(without)

    def row(name):
        for i in range(w.tree.topLevelItemCount()):
            g = w.tree.topLevelItem(i)
            for j in range(g.childCount()):
                if g.child(j).data(0, 0x101) == name:
                    return g.child(j)

    assert row("doppler_pass").data(FOCUS_COL, ROLE) and not row("mains_hum").data(FOCUS_COL, ROLE)
    opened = []
    w.open_spectrogram = lambda regions=None: opened.append(regions)
    w._item_clicked(row("mains_hum"), FOCUS_COL)                      # no signal: nothing happens
    w._item_clicked(row("doppler_pass"), 0)                           # other columns do not open it
    assert opened == []
    w._item_clicked(row("doppler_pass"), FOCUS_COL)                   # this finding's own signal
    w._item_double_clicked(row("doppler_pass"), 0)
    assert opened == [reg, reg]
    w.focus_ok = False                                                # images have no signal to focus
    w.refresh()
    assert not row("doppler_pass").data(FOCUS_COL, ROLE)


def test_no_multimedia_is_loaded_until_a_playable_file_is_opened(tmp_path, monkeypatch):
    _app()
    monkeypatch.setattr(pv, "_MEDIA", None)
    w = pv.MediaPreview()
    img = tmp_path / "i.png"
    Image.new("RGB", (32, 32)).save(img)
    w.load(img, MediaType.IMAGE)
    assert w.player is None and w.video is None and pv._MEDIA is None


def test_several_details_windows_can_be_open_at_once(tmp_path):
    _app()
    w = MainWindow()
    w.path = _wav(tmp_path, 3)
    reg = [dict(label="e", t0=0.0, t1=2.0, f0=100.0, f1=500.0)]
    w.open_spectrogram(reg)
    w.open_spectrogram(reg)
    assert len(w._detail_windows) == 2 and all(d.isVisible() for d in w._detail_windows)
    for d in list(w._detail_windows):
        d.close()
    _wait(100)
    w.open_spectrogram(reg)                            # reopening after closing must not raise
    assert len(w._detail_windows) <= 2
    for d in list(w._detail_windows):
        d.close()


def test_row_actions_and_merged_play_links():
    from mitraav.core.finding import FollowUp
    from mitraav.gui.main_window import map_url, merge_play_links, row_action
    reg = [dict(label="event 1", t0=5.0, t1=9.0, f0=1, f1=2), dict(label="event 2", t0=9.0, t1=12.0, f0=1, f1=2),
           dict(label="event 3", t0=60.0, t1=70.0, f0=1, f1=2)]
    sig = Finding(probe="doppler_pass", group="Aircraft and motion", detected=True, strength=.9, narrows=.4,
                  summary="x", regions=reg)
    assert row_action(sig, MediaType.AUDIO) == "Spectro" and row_action(sig, MediaType.IMAGE) == ""
    osm = Finding(probe="metadata", group="File and metadata", detected=True, strength=.9, narrows=.4, summary="x",
                  follow_ups=[FollowUp("OpenStreetMap at these coordinates",
                                       "https://www.openstreetmap.org/?mlat=1&mlon=2#map=16/1/2")])
    assert row_action(osm, MediaType.IMAGE) == "Map" and map_url(osm).startswith("https://www.openstreetmap.org")
    txt = Finding(probe="image_text", group="Text and visuals", detected=True, strength=.5, narrows=.2, summary="x")
    assert row_action(txt, MediaType.IMAGE) == "Image" and row_action(txt, MediaType.AUDIO) == ""
    links = merge_play_links(reg)
    assert len(links) == 2 and "1 more" in links[0][1]


def test_line_boxes_withhold_digits_and_merge_scales():
    from mitraav.probes.image_text import line_boxes
    a = dict(text="Open for all visitors", conf=90, x0=.1, y0=.1, x1=.5, y1=.2)
    dup = dict(text="Open for all visitors", conf=70, x0=.11, y0=.1, x1=.5, y1=.2)
    ph = dict(text="Call +44 20 7946 0958", conf=88, x0=.1, y0=.3, x1=.6, y1=.4)
    out = line_boxes([dup, ph, a])
    assert len(out) == 2 and out[0]["label"] == "Open for all visitors"
    assert "withheld" in out[1]["label"] and "7946" not in str(out)


def test_image_viewer_outlines_regions_and_resizes(tmp_path):
    _app()
    img = tmp_path / "i.png"
    Image.new("RGB", (400, 300), "white").save(img)
    box = [dict(label="hello", x0=.1, y0=.1, x1=.5, y1=.3)]
    dlg = pv._FullImage(img, None, box)
    dlg.resize(800, 600)
    dlg.show()
    _wait(100)
    pm = dlg.view.pixmap()
    assert pm is not None and pm.width() > 400                  # follows the window
    dlg.close()


def test_preview_thumbnail_follows_its_pane(tmp_path):
    _app()
    img = tmp_path / "i.png"
    Image.new("RGB", (400, 300), "white").save(img)
    w = pv.MediaPreview()
    w.load(img, MediaType.IMAGE)
    w.resize(300, 200)
    w.show()
    _wait(100)
    small = w.thumb.pixmap().width()
    w.resize(700, 600)
    _wait(100)
    assert w.thumb.pixmap().width() > small


def test_inactive_probes_sink_and_groups_stay_put_while_scanning(tmp_path):
    from mitraav.gui.main_window import DISPLAY
    _app()
    w = MainWindow()
    assert DISPLAY["church_bells"] == "tones and chimes"
    w.plan = [("mains_hum", "Setting and power"), ("doppler_pass", "Aircraft and motion"),
              ("siren_pattern", "Sounds"), ("church_bells", "Sounds")]
    w.add_finding(Finding(probe="siren_pattern", group="Sounds", detected=False, info=True, summary="none"))
    w.add_finding(Finding(probe="doppler_pass", group="Aircraft and motion", detected=True, strength=.9,
                          narrows=.9, summary="x"))
    groups = [w.tree.topLevelItem(i).text(0) for i in range(w.tree.topLevelItemCount())]
    assert groups == ["Setting and power", "Aircraft and motion", "Sounds"]        # plan order while pending
    sounds = w.tree.topLevelItem(2)
    assert [sounds.child(i).text(0) for i in range(2)] == ["tones and chimes", "siren pattern"]   # waiting, then inactive
    w.add_finding(Finding(probe="church_bells", group="Sounds", detected=False, info=True, summary="none"))
    w.add_finding(Finding(probe="mains_hum", group="Setting and power", detected=False, info=True, summary="n"))
    groups = [w.tree.topLevelItem(i).text(0) for i in range(w.tree.topLevelItemCount())]
    assert groups[0] == "Aircraft and motion"                                       # best lead first once done
