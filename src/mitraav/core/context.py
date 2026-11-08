"""Shared context: what earlier probes learned, so later probes can use it.

A probe publishes facts in `Finding.meta["context"]` (small, picklable dicts keyed by topic); the
runner merges them and hands the result to every later probe as `self.context`. Probes only ever
*read* it, and must work without it (a partial run, or an add-on that is not installed, leaves keys out).

Topics in use:
  sound   {"busy": share of windows that are speech or music, "shares": {category: share}}   (sound_events)
  scene   {"path", "kind", "label", "p", "kinds"}   whole-picture scene (scene_type, images only)
  text    {"language", "script", "regions"}          readable text clues (image_text)
"""
from __future__ import annotations


def merge(ctx: dict, new: dict | None) -> dict:
    """Return ctx updated with a probe's published topics (later results replace earlier ones)."""
    out = dict(ctx)
    for topic, value in (new or {}).items():
        out[topic] = value
    return out


def from_findings(findings) -> dict:
    """The context an earlier scan produced, rebuilt from its findings (for a partial rerun)."""
    ctx: dict = {}
    for f in findings:
        ctx = merge(ctx, f.meta.get("context"))
    return ctx
