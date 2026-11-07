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
