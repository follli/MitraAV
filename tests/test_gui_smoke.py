import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from mitraav.core.finding import Finding
from mitraav.gui.main_window import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _group_names(w):
    return [w.tree.topLevelItem(i).text(0) for i in range(w.tree.topLevelItemCount())]


def _children(w, group):
    for i in range(w.tree.topLevelItemCount()):
        g = w.tree.topLevelItem(i)
        if g.text(0) == group:
            return [g.child(j).text(0) for j in range(g.childCount())]
    return []


def test_window_nests_findings_by_group_and_sorts_leads_first():
    _app()
    w = MainWindow()
    w.add_finding(Finding(probe="a", group="Sounds", detected=False))
    w.add_finding(Finding(probe="b", group="Sounds", detected=True, strength=0.5, narrows=0.5, summary="s"))
    w.add_finding(Finding(probe="video_encode", group="File and metadata", detected=True,
                          strength=0.45, narrows=0.12))
    assert _group_names(w) == ["Sounds", "File and metadata"]       # file history always last
    assert _children(w, "Sounds") == ["b", "a"]                      # the lead sorts first


def test_running_and_queued_rows_then_result_replaces_them():
    _app()
    w = MainWindow()
    w.plan = [("one", "Sounds"), ("two", "Sounds")]
    w.running = "one"
    w.refresh()
    g = w.tree.topLevelItem(0)
    assert g.child(0).text(1).startswith("running") and g.child(1).text(1) == "queued"
    w.add_finding(Finding(probe="one", group="Sounds", detected=False))
    w.running = "two"
    w.refresh()
    g = w.tree.topLevelItem(0)                      # refresh rebuilds the tree
    kids = {g.child(i).text(0): g.child(i).text(1) for i in range(g.childCount())}
    assert kids["one"] == "info" and kids["two"].startswith("running")


def test_labels_and_very_low_flag():
    faint = Finding(probe="image_text", group="Text and visuals", detected=True, strength=0.25, narrows=0.3)
    assert faint.very_low and faint.confidence_label == "faint" and faint.narrows_label == "a lot"
    hist = Finding(probe="video_encode", group="File and metadata", detected=True, strength=0.45,
                   narrows=0.12)
    assert not hist.very_low and hist.narrows_label == "file history only"
    assert hist.confidence_label == "likely"
    clear = Finding(probe="x", group="Sounds", detected=True, strength=0.9, narrows=0.5)
    assert clear.confidence_label == "clear" and not clear.very_low
    assert Finding(probe="y", detected=False).confidence_label == "-"


def test_detail_shows_very_low_note():
    _app()
    w = MainWindow()
    w.add_finding(Finding(probe="image_text", group="Text and visuals", detected=True,
                          strength=0.25, narrows=0.3, summary="s"))
    w.tree.topLevelItem(0).child(0).setSelected(True)
    assert "Very low lead" in w.detail.toHtml()


def test_quiet_context_is_collapsed_unflagged_and_not_a_lead():
    _app()
    w = MainWindow()
    w.add_finding(Finding(probe="scene_type", group="Visual context", quiet=True, detected=True,
                          strength=0.8, narrows=0.05, summary="picnic area"))
    w.add_finding(Finding(probe="mains_hum", group="Setting and power", detected=True, strength=0.5,
                          narrows=0.4, summary="50 Hz"))
    w.add_finding(Finding(probe="video_encode", group="File and metadata", detected=True,
                          strength=0.45, narrows=0.12))
    assert _group_names(w) == ["Setting and power", "Visual context", "File and metadata"]
    ctx = w.tree.topLevelItem(1)
    assert ctx.text(1) == "context only" and not ctx.isExpanded()
    assert ctx.child(0).text(1) == "context" and ctx.child(0).text(2) == "visible to the eye"
    assert not Finding(probe="x", group="Visual context", quiet=True, detected=True,
                       strength=0.2, narrows=0.05).very_low
