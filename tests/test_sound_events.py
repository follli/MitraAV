import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mitraav.core.models import yamnet_path
from mitraav.probes import sound_events as se

CLIP = Path(__file__).resolve().parents[1] / "clip_sample.wav"
LABELS = ["Speech", "Train horn", "Church bell", "Siren", "Silence", "Environmental noise",
          "Cricket", "Gunshot, gunfire", "Outside, rural or natural"]


def _scores(**hot):
    s = np.zeros(len(LABELS), dtype=np.float32)
    names = {"Train_horn": "Train horn", "Church_bell": "Church bell", "Siren": "Siren",
             "Gunshot_gunfire": "Gunshot, gunfire"}
    for k, v in hot.items():
        s[LABELS.index(names[k])] = v
    return s


def test_categorise():
    assert se.categorise("Train horn") == "rail" and se.categorise("Church bell") == "bells"
    assert se.categorise("Siren") == "sirens" and se.categorise("Cricket") == "animals"
    assert se.categorise("Silence") is None and se.categorise("Environmental noise") is None
    assert se.categorise("Outside, rural or natural") == "setting"


def test_aggregate_thresholds_and_dedupe():
    w = [(float(i), _scores(Train_horn=0.5)) for i in range(4)]          # 4 windows of train horn
    w += [(0.0, _scores(Train_horn=0.5))]                                  # duplicate from chunk overlap
    w += [(10.0, _scores(Siren=0.9))]                                      # one strong window
    w += [(20.0, _scores(Church_bell=0.4))]                                # one weak window
    ev = {e["label"]: e for e in se.summarise(se.aggregate(w, LABELS))}
    assert ev["Train horn"]["n"] == 4 and ev["Siren"]["best"] > 0.89
    assert "Church bell" not in ev


def test_loud_events_need_two_windows():
    w = [(0.0, _scores(Gunshot_gunfire=0.5)), (5.0, _scores(Gunshot_gunfire=0.5))]
    assert se.summarise(se.aggregate(w, LABELS))[0]["cat"] == "impulsive"
    assert not se.summarise(se.aggregate(w[:1], LABELS))


def test_not_run_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "yamnet_path", lambda: tmp_path / "missing.tflite")
    f = se.SoundEventsProbe().run(tmp_path / "x.wav")
    assert f.status == "not_run" and "mitraav-setup" in f.detail


@pytest.mark.skipif(not CLIP.exists() or not shutil.which("ffmpeg") or not yamnet_path().exists()
                    or importlib.util.find_spec("ai_edge_litert") is None,
                    reason="needs clip_sample.wav, ffmpeg, the YAMNet model and ai-edge-litert")
def test_real_clip_names_nature(tmp_path):
    out = tmp_path / "w.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "1161", "-t", "160", "-i", str(CLIP), "-ac", "1",
                    "-ar", "16000", str(out)], check=True)
    f = se.SoundEventsProbe().run(out)
    assert f.detected and "cricket" in f.summary.lower() and "bird" in f.summary.lower()
