import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from mitraav.core import history
from mitraav.gui.history_dialog import HistoryDialog


def test_dialog_lists_names_and_returns_selection(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("MITRAAV_HISTORY", str(tmp_path / "h.json"))
    a, b = tmp_path / "alpha.wav", tmp_path / "beta.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    history.add(a, 2, 10)
    history.add(b, 0, 6)
    d = HistoryDialog()
    texts = [d.list.item(i).text() for i in range(d.list.count())]
    assert texts[0].startswith("beta.jpg") and texts[1].startswith("alpha.wav")
    d.list.setCurrentRow(1)
    d.scan_again()
    assert d.selected_path == a


def test_empty_history_placeholder(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("MITRAAV_HISTORY", str(tmp_path / "none.json"))
    d = HistoryDialog()
    assert d.list.count() == 1 and d.list.item(0).text() == "No scans yet"
    assert not d.scan_btn.isEnabled()
