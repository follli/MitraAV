import importlib.util

import numpy as np
import pytest
from PIL import Image

from mitraav.core.finding import plain
from mitraav.core.models import places_path
from mitraav.probes import scene_type as st
from mitraav.probes.reverse_search import ReverseSearchProbe, dhash, distance


def _img(seed=0, size=(160, 120)):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8"))


def test_labels_load_and_kinds():
    names, io = st.load_labels()
    assert len(names) == 365 == len(io) and set(io) == {"indoor", "outdoor"}
    assert st.kind_of("church/outdoor") == "worship" and st.kind_of("street") == "street"
    assert st.kind_of("rice_paddy") == "nature" and st.kind_of("castle") == "landmark"
    assert st.kind_of("kitchen") is None
    assert st.readable("church/outdoor") == "church (outdoor)" and st.readable("rice_paddy") == "rice paddy"


def test_scene_probe_is_quiet_and_not_run_without_model(tmp_path, monkeypatch):
    p = tmp_path / "a.png"
    _img().save(p)
    monkeypatch.setattr(st, "places_path", lambda: tmp_path / "missing.tflite")
    f = st.SceneTypeProbe().run(p)
    assert f.status == "not_run" and "mitraav-setup" in f.detail
    assert st.SceneTypeProbe.quiet and st.SceneTypeProbe.group == "Visual context"


@pytest.mark.skipif(importlib.util.find_spec("ai_edge_litert") is None or not places_path().exists(),
                    reason="needs ai-edge-litert and the Places365 model")
def test_scene_probe_runs_and_points_to_references(tmp_path):
    # a flat grey-green field with a bright sky band: any plausible outdoor label is fine,
    # what matters is that the pipeline returns a calibrated, quiet finding
    arr = np.zeros((240, 320, 3), np.uint8)
    arr[:100] = (170, 200, 235)
    arr[100:] = (70, 130, 60)
    p = tmp_path / "field.png"
    Image.fromarray(arr).save(p)
    f = st.SceneTypeProbe().run(p)
    assert f.status == "ok" and f.detected and ("the scene looks like" in plain(f.summary) or "unsure" in plain(f.summary)) and f.narrows <= 0.05


def test_dhash_stable_under_resize_and_distinct_otherwise():
    a = _img(1, (320, 240))
    small = a.resize((160, 120))
    assert distance(dhash(a), dhash(small)) <= 10
    assert distance(dhash(a), dhash(_img(2, (320, 240)))) > 15


def test_reverse_search_gives_path_hash_and_links(tmp_path):
    p = tmp_path / "pic.png"
    _img(3, (400, 300)).save(p)
    f = ReverseSearchProbe().run(p)
    assert f.detected and str(p) in f.summary and "Perceptual hash" in f.summary
    assert any("lens.google.com" in u.url for u in f.follow_ups) and ReverseSearchProbe.quiet
