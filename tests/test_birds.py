import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from mitraav.core import birdworker
from mitraav.probes import birds
from mitraav.probes.birds import BirdProbe, summarise

CLIP = Path(__file__).resolve().parents[1] / "clip_sample.wav"


def test_summarise_filters_and_dedupes():
    d = [dict(t0=3.0, t1=6.0, sci="A a", common="A", conf=0.95),
         dict(t0=3.1, t1=6.1, sci="A a", common="A", conf=0.95),     # duplicate from overlap
         dict(t0=9.0, t1=12.0, sci="B b", common="B", conf=0.6),
         dict(t0=12.0, t1=15.0, sci="B b", common="B", conf=0.7),
         dict(t0=20.0, t1=23.0, sci="C c", common="C", conf=0.6)]    # single weak
    s = summarise(d)
    assert [x["sci"] for x in s] == ["B b", "A a"] or [x["sci"] for x in s] == ["A a", "B b"]
    assert next(x for x in s if x["sci"] == "A a")["n"] == 1
    assert all(x["sci"] != "C c" for x in s)


def test_not_run_without_package(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    f = BirdProbe().run(tmp_path / "x.wav")
    assert f.status == "not_run" and "Add-ons" in f.detail and "--install birds" in f.detail


def test_not_run_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(birds, "model_path", lambda: tmp_path / "missing.tflite")
    f = BirdProbe().run(tmp_path / "x.wav")
    assert f.status == "not_run" and "never download" in f.detail


@pytest.mark.skipif(not CLIP.exists() or not shutil.which("ffmpeg")
                    or importlib.util.find_spec("birdnet") is None
                    or not (birdworker.model_path() and birdworker.model_path().exists()),
                    reason="needs clip_sample.wav, ffmpeg, birdnet and its model")
def test_real_clip_has_corvid(tmp_path):
    out = tmp_path / "w.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "1161", "-t", "160", "-i", str(CLIP),
                    "-ac", "1", "-ar", "48000", str(out)], check=True)
    f = BirdProbe().run(out)
    assert f.detected and "Corvus" in f.summary and "region" in f.constrains
    assert any("xeno-canto" in u.url for u in f.follow_ups)
