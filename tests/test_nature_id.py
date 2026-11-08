import sys
import types

import pytest
from PIL import Image

from mitraav.core.finding import plain
from mitraav.probes import nature_id as ni


def _fake_bioclip(preds):
    mod = types.ModuleType("bioclip")

    class Rank:
        SPECIES = 7

    class TreeOfLifeClassifier:
        def __init__(self, **kw):
            pass

        def predict(self, path, rank, k=5):
            return preds

    mod.Rank, mod.TreeOfLifeClassifier = Rank, TreeOfLifeClassifier
    return mod


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "leaf.png"
    Image.new("RGB", (64, 64), "green").save(p)
    return p


def _ready(monkeypatch, preds):
    monkeypatch.setattr(ni, "installed", lambda: True)
    monkeypatch.setattr(ni, "files_present", lambda: True)
    monkeypatch.setitem(sys.modules, "bioclip", _fake_bioclip(preds))


def test_not_run_without_package(img, monkeypatch):
    monkeypatch.setattr(ni, "installed", lambda: False)
    f = ni.NatureIdProbe().run(img)
    assert f.status == "not_run" and "Add-ons" in f.detail and "--install nature" in f.detail


def test_not_run_without_files_never_downloads(img, monkeypatch):
    monkeypatch.setattr(ni, "installed", lambda: True)
    monkeypatch.setattr(ni, "files_present", lambda: False)
    f = ni.NatureIdProbe().run(img)
    assert f.status == "not_run" and "never download" in f.detail and "--install nature" in f.detail


def test_species_lead_with_range_links(img, monkeypatch):
    _ready(monkeypatch, [
        dict(kingdom="Plantae", genus="Quercus", species="Quercus rubra", common_name="northern red oak", score=0.82),
        dict(kingdom="Plantae", genus="Quercus", species="Quercus coccinea", common_name="scarlet oak", score=0.09)])
    f = ni.NatureIdProbe().run(img)
    assert f.detected and "Quercus rubra" in f.summary and "consistent with" in f.summary
    urls = " ".join(u.url for u in f.follow_ups)
    assert "gbif.org" in urls and "inaturalist.org" in urls and "region" in f.constrains


def test_genus_fallback_when_species_unclear(img, monkeypatch):
    _ready(monkeypatch, [
        dict(kingdom="Plantae", genus="Acer", species="Acer rubrum", score=0.30),
        dict(kingdom="Plantae", genus="Acer", species="Acer saccharum", score=0.25),
        dict(kingdom="Plantae", genus="Acer", species="Acer platanoides", score=0.12)])
    f = ni.NatureIdProbe().run(img)
    assert f.detected and "genus Acer" in plain(f.summary) and f.narrows <= 0.2


def test_people_are_never_reported(img, monkeypatch):
    _ready(monkeypatch, [dict(kingdom="Animalia", genus="Homo", species="Homo sapiens", score=0.95)])
    f = ni.NatureIdProbe().run(img)
    assert not f.detected and "Homo" not in f.summary and "people are not analysed" in f.summary


def test_low_scores_not_a_lead(img, monkeypatch):
    _ready(monkeypatch, [dict(kingdom="Plantae", genus="A", species="A a", score=0.2),
                         dict(kingdom="Plantae", genus="B", species="B b", score=0.15)])
    assert not ni.NatureIdProbe().run(img).detected


def test_nature_id_also_runs_on_video_frames():
    from mitraav.probes.nature_id import NatureIdProbe
    assert NatureIdProbe.frames_ok


def _scene_by_region(monkeypatch, whole, regions):
    """Scene model stub: `whole` for the picture, regions[k] for region{k}.png (default: a building)."""
    def fake(path):
        n = path.name
        if n.startswith("region"):
            return regions.get(int(n[6]), ("landmark", "tower", 0.8))
        return whole
    monkeypatch.setattr(ni, "_scene_of", fake)


def _ready_by_file(monkeypatch, by_name):
    monkeypatch.setattr(ni, "installed", lambda: True)
    monkeypatch.setattr(ni, "files_present", lambda: True)
    mod = _fake_bioclip([])
    mod.TreeOfLifeClassifier.predict = lambda self, path, rank, k=5: by_name.get(path.split("/")[-1], by_name["default"])
    monkeypatch.setitem(sys.modules, "bioclip", mod)


def test_a_building_scene_does_not_hide_a_plant_in_one_region(tmp_path, monkeypatch):
    big = tmp_path / "cathedral.png"
    Image.new("RGB", (400, 300), "gray").save(big)
    _scene_by_region(monkeypatch, ("worship", "cathedral", 0.7), {1: ("nature", "forest", 0.6)})
    ginkgo = [dict(kingdom="Plantae", genus="Ginkgo", species="Ginkgo biloba", score=0.58)]     # whole-picture hallucination
    oak = [dict(kingdom="Plantae", genus="Quercus", species="Quercus robur", common_name="English oak", score=0.8)]
    _ready_by_file(monkeypatch, {"default": ginkgo, "region1.png": oak})
    f = ni.NatureIdProbe().run(big)
    assert f.detected and "Quercus robur" in f.summary and "upper right" in f.summary
    assert "Ginkgo" not in f.summary                         # the building-scene whole picture is not trusted
    assert f.boxes and f.boxes[0]["x0"] == 0.5 and f.boxes[0]["y1"] == 0.5


def test_a_building_scene_with_nothing_natural_anywhere_stays_an_info_message(tmp_path, monkeypatch):
    big = tmp_path / "cathedral.png"
    Image.new("RGB", (400, 300), "gray").save(big)
    _scene_by_region(monkeypatch, ("worship", "cathedral", 0.7), {})
    _ready_by_file(monkeypatch, {"default": [dict(kingdom="Plantae", genus="Ginkgo", species="Ginkgo biloba", score=0.58)]})
    f = ni.NatureIdProbe().run(big)
    assert not f.detected and f.info and "building or street scene" in f.summary


def test_regions_need_a_higher_score_than_the_whole_picture(tmp_path, monkeypatch):
    big = tmp_path / "x.png"
    Image.new("RGB", (400, 300), "gray").save(big)
    _scene_by_region(monkeypatch, ("nature", "forest", 0.7), {k: ("nature", "forest", 0.7) for k in range(5)})
    weak = [dict(kingdom="Plantae", genus="A", species="A a", score=0.55)]               # passes for a whole picture only
    _ready_by_file(monkeypatch, {"default": weak})
    f = ni.NatureIdProbe().run(big)
    assert f.detected and "upper" not in f.summary and "lower" not in f.summary and not f.boxes


def test_scene_from_shared_context_is_used_for_the_same_picture(tmp_path, monkeypatch):
    big = tmp_path / "c.png"
    Image.new("RGB", (400, 300), "gray").save(big)
    calls = []
    monkeypatch.setattr(ni, "_scene_of", lambda p: calls.append(p.name) or ("nature", "forest", 0.7))
    _ready_by_file(monkeypatch, {"default": [dict(kingdom="Plantae", genus="Q", species="Q r", score=0.9)]})
    probe = ni.NatureIdProbe()
    probe.context = {"scene": dict(path=str(big), kind="worship", label="church", p=0.8, kinds=["worship"])}
    f = probe.run(big)
    assert "c.png" not in calls and f.detected                 # whole-picture scene came from the context


def test_a_scan_forces_huggingface_offline_before_anything_is_imported(img, monkeypatch):
    import os
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(ni, "installed", lambda: False)
    ni.NatureIdProbe().run(img)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_a_species_seen_in_a_single_video_frame_is_only_unconfirmed(tmp_path, monkeypatch):
    from mitraav.core import runner
    from mitraav.core.finding import Finding
    frames = [(float(i), tmp_path / f"f{i}.jpg") for i in range(5)]
    monkeypatch.setattr(runner, "extract_frames", lambda p: frames)
    hit = Finding(probe="nature_id", detected=True, strength=0.7, narrows=0.3, summary="a **cat**", meta={"label": "Felis catus"})
    miss = Finding(probe="nature_id", detected=False, summary="none")
    results = {frames[1][1]: hit}
    monkeypatch.setattr(runner, "run_probe", lambda cls, p, ctx=None: results.get(p, miss))
    f = runner.run_on_frames(ni.NatureIdProbe, tmp_path / "v.mp4")
    assert not f.detected and f.info and "only 1 of 5" in f.summary and "cat" in f.summary
    for i in (1, 3):
        results[frames[i][1]] = hit
    f = runner.run_on_frames(ni.NatureIdProbe, tmp_path / "v.mp4")
    assert f.detected and "2/5 sampled frames" in f.summary
