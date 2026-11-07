import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PySide6.QtWidgets import QApplication

from mitraav.core import addons
from mitraav.setup_models import DEFAULTS


def test_registry_has_the_four_addons_and_bioclip_is_heavy_and_last():
    keys = [a.key for a in addons.registry()]
    assert keys == ["sounds", "scene", "birds", "nature"]
    nature = addons.by_key("nature")
    assert nature.heavy and "1.7 GB" in nature.size
    first = nature.steps[0].command()
    assert "torch" in first and "torchvision" in first and "--index-url" in first
    assert "download.pytorch.org/whl/cpu" in " ".join(first)                  # never the CUDA build


def test_bioclip_never_in_the_defaults():
    assert "nature" not in DEFAULTS and set(DEFAULTS) <= {"sounds", "scene"}


def test_status_logic():
    state = {"a": False, "b": False}
    a = addons.Addon("x", "X", "adds", "1 MB", [
        addons.Step("one", lambda: state["a"], ["true"]), addons.Step("two", lambda: state["b"], ["true"])])
    assert a.status() == "not installed" and len(a.pending()) == 2
    state["a"] = True
    assert a.status() == "partly installed" and [s.label for s in a.pending()] == ["two"]
    state["b"] = True
    assert a.status() == "installed" and not a.pending()


def test_run_steps_streams_output_and_stops_on_failure():
    lines = []
    ok = addons.Addon("ok", "Ok", "", "", [addons.Step("echo", lambda: False,
                                                       [sys.executable, "-c", "print('hello'); print('world')"])])
    addons.run_steps(ok, lines.append)
    assert "hello" in lines and "world" in lines and lines[-1].startswith("Done")
    bad = addons.Addon("bad", "Bad", "", "", [addons.Step("fail", lambda: False,
                                                          [sys.executable, "-c", "import sys; sys.exit(3)"])])
    with pytest.raises(RuntimeError, match="exit code 3"):
        addons.run_steps(bad, lambda t: None)


def test_dialog_lists_addons_and_hides_install_when_installed(monkeypatch):
    QApplication.instance() or QApplication([])
    from mitraav.gui.addons_dialog import AddonsDialog
    monkeypatch.setattr(addons.Addon, "status", lambda self: "installed" if self.key != "nature" else "not installed")
    d = AddonsDialog()
    d.show()
    assert set(d.rows) == {"sounds", "scene", "birds", "nature"}
    assert d.rows["nature"][1].isVisibleTo(d) and d.rows["nature"][1].text() == "Install"
    assert not d.rows["sounds"][1].isVisibleTo(d)
    assert "installed" in d.rows["sounds"][0].text()


def test_install_asks_for_confirmation_and_does_nothing_on_no(monkeypatch):
    QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QMessageBox

    from mitraav.gui import addons_dialog
    monkeypatch.setattr(addons.Addon, "status", lambda self: "not installed")
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    d = addons_dialog.AddonsDialog()
    d.install("nature")
    assert asked and "1.7 GB" in asked[0] and "large download" in asked[0] and d.worker is None


def test_torch_pair_check_and_repair_command(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(sys, "platform", "linux")
    ver = {"torch": "2.15.0+cpu", "torchvision": "0.29.2"}          # the mismatch seen on a real install
    monkeypatch.setattr(metadata, "version", lambda n: ver[n])
    assert not addons._torch_ok()
    ver["torchvision"] = "0.30.0+cpu"
    assert addons._torch_ok()
    monkeypatch.setattr(addons, "_has", lambda m: True)
    cmd = addons._torch_command()
    assert "--force-reinstall" in cmd and "--no-deps" in cmd and addons.CPU_INDEX in cmd
    monkeypatch.setattr(addons, "_has", lambda m: False)
    assert "--force-reinstall" not in addons._torch_command()        # fresh install: normal resolution


def test_close_works_after_a_failed_install(monkeypatch):
    QApplication.instance() or QApplication([])
    from mitraav.gui import addons_dialog
    monkeypatch.setattr(addons.Addon, "status", lambda self: "not installed")
    d = addons_dialog.AddonsDialog()
    d._busy = True
    d.refresh()
    assert not d.close_btn.isEnabled()
    d._finished(False, "step failed")                                # what the worker signals
    assert d.close_btn.isEnabled() and d.rows["nature"][1].isEnabled()
