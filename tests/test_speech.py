import types

import numpy as np
from scipy.io import wavfile

from mitraav.core import speech
from mitraav.probes import speech_language as sl


def _fake_whisper(monkeypatch, lang="de", prob=0.93, segs=((0.0, 5.0, "Das ist ein Test am 2024-06-01 in der Stadt"),)):
    class Model:
        def transcribe(self, audio, **kw):
            items = [types.SimpleNamespace(start=a, end=b, text=t) for a, b, t in segs]
            return iter(items), types.SimpleNamespace(language=lang, language_probability=prob)

    monkeypatch.setattr(speech, "installed", lambda: True)
    monkeypatch.setattr(speech, "files_present", lambda: True)
    monkeypatch.setattr(speech, "offline_model", lambda: Model())


def _wav(tmp_path, secs=8):
    p = tmp_path / "a.wav"
    wavfile.write(p, 16000, (np.random.default_rng(0).standard_normal(16000 * secs) * 3000).astype(np.int16))
    return p


def test_language_and_clues_are_reported_without_the_transcript(tmp_path, monkeypatch):
    _fake_whisper(monkeypatch)
    f = sl.SpeechLanguageProbe().run(_wav(tmp_path))
    assert f.detected and "German" in f.summary and "93%" in f.summary and "2024-06-01" in f.summary
    assert "Das ist ein Test" not in f.summary and f.meta["geo"]["any_of"] == ["europe"]


def test_low_confidence_is_called_a_guess_and_gives_no_geo_hint(tmp_path, monkeypatch):
    _fake_whisper(monkeypatch, lang="ja", prob=0.41, segs=((0.0, 6.0, "a b c d e f"),))
    f = sl.SpeechLanguageProbe().run(_wav(tmp_path))
    assert "unsure" in f.summary and not f.meta.get("geo")


def test_no_speech_and_missing_addon_are_handled(tmp_path, monkeypatch):
    _fake_whisper(monkeypatch, segs=((0.0, 1.0, "hm"),))
    assert not sl.SpeechLanguageProbe().run(_wav(tmp_path)).detected
    monkeypatch.setattr(speech, "installed", lambda: False)
    f = sl.SpeechLanguageProbe().run(_wav(tmp_path))
    assert f.status == "not_run" and "--install speech" in f.detail


def test_scan_goes_offline_before_importing_the_hub(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(speech, "installed", lambda: False)
    sl.SpeechLanguageProbe().run(_wav(tmp_path))
    assert os.environ["HF_HUB_OFFLINE"] == "1"
