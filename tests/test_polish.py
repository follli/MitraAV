import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from scipy.io import wavfile

from mitraav.core import birdworker
from mitraav.core.finding import Finding, plain
from mitraav.gui import theme
from mitraav.gui.main_window import _badge, _rich
from mitraav.probes.audio_provenance import AudioProvenanceProbe
from mitraav.probes.jpeg_forensics import JpegForensicsProbe

C = theme.COLORS


def _det(sci, common, t, conf):
    return dict(sci=sci, common=common, t0=t, t1=t + 3, conf=conf)


def test_birdnet_non_bird_labels_are_never_species():
    dets = [_det("Human vocal", "Human vocal", 3.0 * i, 0.9) for i in range(6)]
    dets += [_det("Turdus merula", "Eurasian Blackbird", 3.0 * i, 0.9) for i in range(3)]
    sp = birdworker.summarise(dets)
    assert [s["sci"] for s in sp] == ["Turdus merula"]
    assert birdworker.human_segments(dets) == 6
    assert birdworker.summarise([_det("Dog", "Dog", 0, 0.99), _det("Dog", "Dog", 3, 0.99)]) == []


def test_pill_colours_and_labels():
    nothing = _badge(Finding(probe="a", detected=False))
    info = _badge(Finding(probe="b", detected=False, info=True))
    lead = _badge(Finding(probe="c", detected=True, strength=0.9, narrows=0.5, group="Sounds"))
    ctx = _badge(Finding(probe="d", detected=True, quiet=True, pill="hash created"))
    inc = _badge(Finding(probe="e", detected=True, strength=0.9, narrows=0.5, caveat="busy"))
    assert nothing == info == ("info", C["blue"])           # finished-with-nothing looks like info
    assert lead == ("lead", C["green"]) and ctx[0] == "hash created" and inc == ("inconclusive", C["amber"])
    assert len({info[1], lead[1], inc[1], ctx[1]}) == 4     # lead, info, inconclusive, context all differ


def test_inconclusive_is_not_a_lead_or_very_low():
    f = Finding(probe="x", group="Sounds", detected=True, strength=0.2, narrows=0.2, caveat="busy audio")
    assert f.inconclusive and not f.very_low


def test_bold_markup():
    assert _rich("the **YouTube** clip & <b>x</b>") == "the <b>YouTube</b> clip &amp; &lt;b&gt;x&lt;/b&gt;"
    assert plain("event **1** at **5 s**") == "event 1 at 5 s"
    d = Finding(probe="p", detected=True, summary="**Event 1**: ok").to_dict()
    assert d["summary"] == "Event 1: ok"                   # JSON and CLI never show the markers


def test_audio_with_a_clean_history_is_information_not_nothing(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "clean.wav"
    wavfile.write(p, 44100, (0.2 * rng.standard_normal(44100 * 30) * 32767).astype(np.int16))
    f = AudioProvenanceProbe().run(p)
    assert not f.detected and f.info and "Nothing unusual" in f.summary


def test_mpo_photo_is_treated_as_a_jpeg(tmp_path):
    img = Image.fromarray((np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8"))
    p = tmp_path / "phone.jpg"
    img.save(p, "MPO", save_all=True, append_images=[img])
    with Image.open(p) as im:
        assert im.format == "MPO"
    f = JpegForensicsProbe().run(p)
    assert "Not a JPEG" not in f.summary and "MPO" in plain(f.summary)


def test_software_tag_alone_is_not_an_edit(tmp_path):
    img = Image.fromarray((np.random.default_rng(2).random((64, 64, 3)) * 255).astype("uint8"))
    ex = Image.Exif()
    ex[0x0131] = "12.4.1"                                   # what an iPhone writes
    p = tmp_path / "ios.jpg"
    img.save(p, "JPEG", exif=ex.tobytes(), qtables=[[3] * 64, [3] * 64])
    f = JpegForensicsProbe().run(p)
    assert not f.detected and f.info


def test_detail_pane_renders_bold_key_words_and_the_busy_banner():
    from PySide6.QtWidgets import QApplication

    from mitraav.gui.main_window import MainWindow
    QApplication.instance() or QApplication([])
    w = MainWindow()
    w.add_finding(Finding(probe="video_encode", group="File and metadata", detected=True, strength=0.5,
                          narrows=0.1, summary="typical of **YouTube** video"))
    w.add_finding(Finding(probe="church_bells", group="Sounds", detected=True, strength=0.5, narrows=0.3,
                          summary="a bell", caveat="Speech fills about 91% of this recording."))
    for i in range(w.tree.topLevelItemCount()):
        g = w.tree.topLevelItem(i)
        for j in range(g.childCount()):
            g.child(j).setSelected(True)
            w.show_detail()
            html_text = w.detail.toHtml()
            if g.child(j).data(0, 0x101) == "video_encode":
                assert "font-weight" in html_text and "YouTube" in html_text and "**" not in html_text
            if g.child(j).data(0, 0x101) == "church_bells":
                assert "Inconclusive" in html_text and "91%" in html_text
            g.child(j).setSelected(False)


def test_progress_bar_is_blue():
    assert f"QProgressBar::chunk {{ background: {C['blue']}" in theme.QSS
