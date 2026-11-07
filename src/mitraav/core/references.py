from __future__ import annotations

from functools import cache
from importlib import resources

import yaml

from .finding import FollowUp


@cache
def _load(file: str) -> dict:
    p = resources.files("mitraav") / "references" / f"{file}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def follow_ups(file: str, key: str) -> list[FollowUp]:
    return [FollowUp(**item) for item in _load(file).get(key, [])]
