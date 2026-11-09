import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mitraav.core import plugins
from mitraav.core.runner import discover_probes, run_probe, scan

TEMPLATE = Path(plugins.__file__).parent.parent / "templates" / "example_plugin.py"


@pytest.fixture
def folder(tmp_path, monkeypatch):
    d = tmp_path / "plugins"
    d.mkdir()
    monkeypatch.setenv("MITRAAV_PLUGINS", str(d))
    monkeypatch.setenv("MITRAAV_PLUGIN_STATE", str(tmp_path / "state.json"))
    plugins.ERRORS.clear()
    return d


def _wav(tmp_path, secs=6):
    t = np.arange(8000 * secs) / 8000
    x = 0.05 * np.sin(2 * np.pi * 300 * t)
    x[8000 * 3:8000 * 4] *= 8                                   # a louder second at 3 s
    p = tmp_path / "a.wav"
    wavfile.write(p, 8000, (x * 32767).astype(np.int16))
    return p


def _write(folder, name, body):
    p = folder / name
    p.write_text(body)
    return p


GOOD = '''
"""A tiny test plugin."""
from mitraav.plugin import Finding, MediaType, Probe

class Tiny(Probe):
    name = "tiny_plugin"
    group = "Sounds"
    media = (MediaType.AUDIO,)
    def run(self, path):
        return Finding(probe=self.name, detected=True, strength=0.5, narrows=0.1, summary="hello from a plugin")
'''


def test_files_are_listed_but_never_executed_until_enabled(folder):
    marker = folder / "ran.txt"
    _write(folder, "spy.py", GOOD + f'\nopen({str(marker)!r}, "w").write("imported")\n')
    pf = plugins.list_plugins()
    assert [p.path.name for p in pf] == ["spy.py"] and pf[0].status == "disabled"
    assert pf[0].doc == "A tiny test plugin." and pf[0].probes == ["Tiny"]
    assert "tiny_plugin" not in [c.name for c in discover_probes()] and not marker.exists()


def test_enabling_loads_the_probe_and_it_runs_in_its_own_process(folder, tmp_path):
    p = _write(folder, "tiny.py", GOOD)
    plugins.enable(p)
    cls = next(c for c in discover_probes() if c.name == "tiny_plugin")
    f = run_probe(cls, _wav(tmp_path))
    assert f.status == "ok" and f.summary == "hello from a plugin"
    f2 = next(f for f in scan(_wav(tmp_path)) if f.probe == "tiny_plugin")
    assert f2.meta["plugin"] == "tiny.py" and f2.group == "Sounds"


def test_an_edited_file_is_disabled_until_reviewed_and_enabled_again(folder):
    p = _write(folder, "tiny.py", GOOD)
    plugins.enable(p)
    assert plugins.list_plugins()[0].status == "enabled"
    p.write_text(GOOD + "\n# edited\n")
    assert plugins.list_plugins()[0].status == "changed"
    assert "tiny_plugin" not in [c.name for c in discover_probes()]
    assert "changed since you enabled" in plugins.ERRORS[str(p.resolve())]
    plugins.enable(p)
    assert "tiny_plugin" in [c.name for c in discover_probes()]
    plugins.disable(p)
    assert plugins.list_plugins()[0].status == "disabled"


def test_a_broken_or_clashing_plugin_never_breaks_discovery(folder):
    bad = _write(folder, "bad.py", "raise RuntimeError('boom on import')\n")
    syn = _write(folder, "syn.py", "def oops(:\n")
    clash = _write(folder, "clash.py", GOOD.replace("tiny_plugin", "mains_hum"))
    new = _write(folder, "future.py", GOOD.replace("tiny_plugin", "future_api") + "\nMITRAAV_API = 99\n")
    for p in (bad, syn, clash, new):
        plugins.enable(p)
    names = [c.name for c in discover_probes()]
    assert names.count("mains_hum") == 1 and "future_api" not in names
    errs = plugins.ERRORS
    assert "boom on import" in errs[str(bad.resolve())]
    assert "already used" in errs[str(clash.resolve())] and "needs plugin API 99" in errs[str(new.resolve())]
    assert any("cannot be read" in p.error for p in plugins.list_plugins() if p.path.name == "syn.py")


def test_a_plugin_that_raises_or_hangs_is_contained(folder, tmp_path):
    body = GOOD.replace('return Finding(probe=self.name, detected=True, strength=0.5, narrows=0.1, summary="hello from a plugin")',
                        'raise ValueError("plugin bug")')
    slow = GOOD.replace("tiny_plugin", "slow_plugin").replace("    media =", "    timeout = 1.0\n    media =").replace(
        'return Finding(probe=self.name, detected=True, strength=0.5, narrows=0.1, summary="hello from a plugin")',
        "import time\n        time.sleep(30)")
    plugins.enable(_write(folder, "raises.py", body))
    plugins.enable(_write(folder, "slow.py", slow))
    by = {c.name: c for c in discover_probes()}
    f = run_probe(by["tiny_plugin"], _wav(tmp_path))
    assert f.status == "error" and "plugin bug" in f.detail and f.probe == "tiny_plugin"
    assert run_probe(by["slow_plugin"], _wav(tmp_path)).status == "timeout"


def test_the_shipped_example_plugin_works(folder, tmp_path):
    p = folder / "loudest.py"
    shutil.copy(TEMPLATE, p)
    plugins.enable(p)
    cls = next(c for c in discover_probes() if c.name == "loudest_moment")
    f = run_probe(cls, _wav(tmp_path))
    assert f.status == "ok" and f.detected and "0:03" in f.summary and f.regions and f.regions[0]["t_play"] >= 0


def test_setup_cli_can_create_enable_and_disable_a_plugin(folder, capsys):
    from mitraav.setup_models import main
    main(["--new-plugin"])
    assert (folder / "loudest_moment.py").exists() and "off until you enable it" in capsys.readouterr().out
    main(["--enable-plugin", "loudest_moment.py"])
    assert plugins.list_plugins()[0].status == "enabled"
    main(["--disable-plugin", "loudest_moment.py"])
    assert plugins.list_plugins()[0].status == "disabled"
    main(["--enable-plugin", "nope.py"])
    assert "No plugin file named 'nope.py'" in capsys.readouterr().out


def test_plugins_dialog_lists_enables_with_confirmation_and_shows_errors(folder, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from mitraav.gui.plugins_dialog import PluginsDialog
    QApplication.instance() or QApplication([])
    p = _write(folder, "tiny.py", GOOD)
    _write(folder, "bad.py", "raise RuntimeError('boom')\n")
    plugins.enable(folder / "bad.py")
    d = PluginsDialog()
    pf = {x.path.name: x for x in plugins.list_plugins()}
    plugins.load_enabled()
    d.refresh()
    from PySide6.QtWidgets import QLabel
    assert any("boom" in lab.text() for lab in d.body.findChildren(QLabel))      # the import error is shown
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    d.toggle(pf["tiny.py"])
    assert plugins.list_plugins()[1].status == "disabled"          # declining enables nothing
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    d.toggle(pf["tiny.py"])
    assert {x.path.name: x.status for x in plugins.list_plugins()}["tiny.py"] == "enabled"
    d.add_example()
    assert (folder / "loudest_moment.py").exists() and p.exists()
