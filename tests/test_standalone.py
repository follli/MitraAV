import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_standalone_doppler.py"
OUT = ROOT / "standalone" / "doppler_standalone.py"


def _build_module():
    spec = importlib.util.spec_from_file_location("build_standalone_doppler", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_standalone_file_is_up_to_date(tmp_path, monkeypatch):
    mod = _build_module()
    fresh = tmp_path / "fresh.py"
    monkeypatch.setattr(mod, "OUT", fresh)
    mod.main()
    assert fresh.read_text() == OUT.read_text(), "run: python scripts/build_standalone_doppler.py"


def test_standalone_file_runs_without_the_package_and_finds_a_glide(tmp_path):
    sr = 16000
    t = np.arange(sr * 30) / sr
    f = 150 + 100 / (1 + np.exp((t - 15) / 2.5))                  # a smooth downward glide, 250 -> 150 Hz
    x = 0.1 * np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.02 * np.random.default_rng(0).standard_normal(t.size)
    wav = tmp_path / "glide.wav"
    wavfile.write(wav, sr, (x * 32767).astype(np.int16))
    # run it from an empty folder with -I: no mitraav on the path, only numpy and scipy
    r = subprocess.run([sys.executable, "-I", str(OUT), str(wav)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "mitraav" not in OUT.read_text().replace("mitraav-cache", "")
    assert "FOUND" in r.stdout or "nothing found" in r.stdout


def test_standalone_reports_a_missing_file_cleanly(tmp_path):
    r = subprocess.run([sys.executable, "-I", str(OUT), str(tmp_path / "nope.wav")], capture_output=True, text=True)
    assert r.returncode != 0 and "not a file" in r.stderr
