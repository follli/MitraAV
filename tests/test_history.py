from pathlib import Path

from mitraav.core import history


def test_add_load_dedupe_and_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("MITRAAV_HISTORY", str(tmp_path / "h.json"))
    assert history.load() == []
    a, b = tmp_path / "a.wav", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    history.add(a, 2, 10)
    history.add(b, 0, 6)
    history.add(a, 3, 10)                       # same file again: moves to the top, no duplicate
    rows = history.load()
    assert [r["name"] for r in rows] == ["a.wav", "b.jpg"] and rows[0]["leads"] == 3
    history.clear()
    assert history.load() == []


def test_damaged_file_is_empty(tmp_path, monkeypatch):
    p = tmp_path / "h.json"
    p.write_text("{not json")
    monkeypatch.setenv("MITRAAV_HISTORY", str(p))
    assert history.load() == []
    history.add(tmp_path / "x.png", 1, 1)       # and it recovers
    assert history.load()[0]["name"] == "x.png"


def test_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MITRAAV_HISTORY", str(tmp_path / "h.json"))
    for i in range(history.MAX_ENTRIES + 5):
        history.add(Path(f"f{i}.wav"), 0, 1)
    assert len(history.load()) == history.MAX_ENTRIES
