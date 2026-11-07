"""Scan history: which files were scanned, when, and how many leads they gave.

Stored as JSON next to the models (see core/models.py). Only the file path, name, time and counts
are kept, never the findings; override the location with MITRAAV_HISTORY (used by the tests).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .models import model_dir

MAX_ENTRIES = 200


def history_path() -> Path:
    env = os.environ.get("MITRAAV_HISTORY")
    return Path(env) if env else model_dir().parent / "history.json"


def load() -> list[dict]:
    """Newest first. A missing or damaged file is an empty history."""
    try:
        data = json.loads(history_path().read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict) and "path" in e]
    except (OSError, ValueError):
        return []


def add(path: Path, leads: int, probes: int) -> None:
    path = Path(path)
    entry = dict(path=str(path.resolve()), name=path.name, leads=int(leads), probes=int(probes),
                 when=datetime.now().isoformat(timespec="seconds"))
    entries = [e for e in load() if e["path"] != entry["path"]]        # one row per file
    entries.insert(0, entry)
    hp = history_path()
    try:
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps(entries[:MAX_ENTRIES], indent=1), encoding="utf-8")
    except OSError:
        pass                                                          # history is a convenience


def clear() -> None:
    try:
        history_path().unlink(missing_ok=True)
    except OSError:
        pass
