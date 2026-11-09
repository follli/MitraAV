import json
from pathlib import Path

from mitraav.core.finding import Finding, FollowUp
from mitraav.core.geo import combine, hint, macros_for_latlon, macros_for_phrase
from mitraav.core.report import location_summary, to_json, to_markdown


def _f(probe, geo=None, **kw):
    kw.setdefault("detected", True)
    kw.setdefault("strength", 0.6)
    kw.setdefault("narrows", 0.3)
    return Finding(probe=probe, group="g", meta={"geo": geo} if geo else {}, **kw)


def test_every_country_phrase_in_the_text_tables_maps_to_a_macro_region():
    import yaml
    d = yaml.safe_load((Path(__file__).parent.parent / "src/mitraav/references/codes.yaml").read_text())
    unmapped = [v for t in ("calling_codes", "cctlds", "currencies") for v in d[t].values() if not macros_for_phrase(v)]
    assert unmapped == []
    assert macros_for_phrase("the US or the UK (miles per hour)") == {"north_america", "europe"}
    assert macros_for_latlon(51.5, -0.12) == {"europe"} and macros_for_latlon(40.7, -74.0) == {"north_america"}


def test_agreeing_hints_share_a_region_and_disagreeing_ones_are_flagged_as_questions():
    gps = hint("GPS tag", {"europe"}, 0.9)
    text = hint("text clues", {"europe"}, 0.3)
    birds = hint("birdsong", {"north_america"}, 0.5)
    ok = location_summary([_f("metadata", gps), _f("image_text", text)])
    assert ok["common"] == ["europe"] and not ok["conflicts"] and "consistent with **Europe**" in ok["notes"][0]
    bad = location_summary([_f("metadata", gps), _f("birdsong", birds)])
    assert bad["conflicts"] == [("metadata", "birdsong")]
    assert "Possible disagreement" in bad["notes"][0] and "Innocent explanations" in bad["notes"][0]


def test_weak_hints_do_not_raise_a_conflict_and_one_hint_is_not_cross_checked():
    weak = hint("text clue", {"asia"}, 0.15)
    strong = hint("GPS tag", {"europe"}, 0.9)
    assert combine([("image_text", weak), ("metadata", strong)])["conflicts"] == []
    one = location_summary([_f("metadata", strong)])
    assert "nothing can be cross-checked" in one["notes"][0]


def test_inconclusive_findings_do_not_contribute_hints():
    f = _f("birdsong", hint("birdsong", {"europe"}, 0.5), caveat="busy audio")
    assert location_summary([f])["n"] == 0


def test_markdown_report_orders_top_leads_then_others_then_nothing_found(tmp_path):
    fs = [_f(f"p{i}", strength=0.9 - i * 0.1, summary=f"lead number {i}") for i in range(5)]
    fs += [Finding(probe="quiet_probe", detected=False, info=True, summary="nothing"),
           Finding(probe="needs_addon", detected=False, status="not_run", detail="install the add-on"),
           _f("busy", caveat="speech fills the recording", summary="x")]
    fs[0] = _f("p0", strength=0.9, summary="the best lead", follow_ups=[FollowUp("A tool", "https://example.org", "note")])
    md = to_markdown(tmp_path / "clip.wav", fs)
    assert md.index("## Top leads") < md.index("## Other leads") < md.index("## Checked, nothing found")
    assert "### p0" in md.split("## Other leads")[0] and "### p3" in md.split("## Other leads")[1]
    assert "[A tool](https://example.org): note" in md and "needs addon" in md and "speech fills the recording" in md
    assert "lead number" in md and "**lead" not in md                       # key-word markers are stripped


def test_json_report_has_location_and_findings(tmp_path):
    data = json.loads(to_json(tmp_path / "clip.wav", [_f("metadata", hint("GPS", {"europe"}, 0.9), summary="s")]))
    assert data["file"] == "clip.wav" and data["location"]["n"] == 1 and data["findings"][0]["probe"] == "metadata"


def test_summary_row_appears_only_when_there_are_location_hints():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from mitraav.gui.main_window import SUMMARY_KEY, MainWindow
    QApplication.instance() or QApplication([])
    w = MainWindow()
    w.add_finding(_f("metadata", hint("GPS", {"europe"}, 0.9)))
    w.add_finding(_f("birdsong", hint("birdsong", {"north_america"}, 0.5)))
    w.refresh()
    top = w.tree.topLevelItem(0)
    assert top.data(0, 0x101) == SUMMARY_KEY and top.text(1) == "check clues"
    top.setSelected(True)
    assert "Possible disagreement" in w.detail.toPlainText()
    w2 = MainWindow()
    w2.add_finding(_f("mains_hum"))
    w2.refresh()
    assert w2.tree.topLevelItem(0).data(0, 0x101) != SUMMARY_KEY
