"""Match stable tone signatures (telephone, emergency broadcast) to a table of known signals.

Only signals with a published dataset to correlate against belong here: the table lives in
references/tones.yaml and every match points at the standard or station list to check next.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.lines import Line, find_lines
from ..core.probe import Probe
from ..core.references import _load, follow_ups

MIN_PROM = 14.0       # dB above the local floor; weaker lines are too easily coincidence
HARMONIC_DB = 35.0     # overtones within this many dB mean a siren or horn, not a pure tone
ARTEFACT_DB = 25.0    # lines this far below the matched tones are treated as artefacts
CADENCE_TOL = 0.35    # allowed relative error on nominal on/off times
DEFAULT_MIN_DUR = 1.5


def _tol(f: float) -> float:
    return max(3.0, 0.012 * f)


def dedupe(lines: list[Line]) -> list[Line]:
    """Overlapping chunks see the same line twice."""
    out: list[Line] = []
    for ln in sorted(lines, key=lambda x: (x.t0, x.f)):
        if not any(abs(ln.f - o.f) < 0.01 * o.f and abs(ln.t0 - o.t0) < 1.0 for o in out):
            out.append(ln)
    return out


def group_simultaneous(lines: list[Line]) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for ln in sorted(lines, key=lambda x: x.t0):
        for g in groups:
            t0, t1 = np.mean([m.t0 for m in g]), np.mean([m.t1 for m in g])
            ov = min(t1, ln.t1) - max(t0, ln.t0)
            if ov >= 0.6 * min(t1 - t0, ln.t1 - ln.t0) and all(abs(ln.f - m.f) > 0.01 * m.f for m in g):
                g.append(ln)
                break
        else:
            groups.append([ln])
    return groups


def _hit(entry: dict, g: list[Line]):
    found = []
    for ef in entry["freqs"]:
        c = [m for m in g if abs(m.f - ef) <= _tol(ef)]
        if not c:
            return None
        found.append(min(c, key=lambda m: abs(m.f - ef)))
    # other simultaneous lines count against the match only if they are within 25 dB of the matched
    # tones: codec artefacts far below the signal are not part of it
    floor = min(m.level for m in found) - ARTEFACT_DB
    extras = sum(1 for m in g if m not in found and m.level >= floor)
    if extras > (0 if len(found) == 1 else 1):
        return None
    return found


def _has_harmonics(line: Line, others: list[Line]) -> bool:
    """A siren or horn has overtones at 2x and 3x; telephone tones are pure sine waves."""
    for m in others:
        if m is line or min(m.t1, line.t1) - max(m.t0, line.t0) < 0.5 * (line.t1 - line.t0):
            continue
        for k in (2, 3):
            if abs(m.f - k * line.f) <= 0.015 * k * line.f and m.level >= line.level - HARMONIC_DB:
                return True
    return False


def match_entry(entry: dict, groups: list[list[Line]]) -> dict | None:
    hits = []
    every = [m for g in groups for m in g]
    for g in groups:
        f = _hit(entry, g)
        if f and entry.get("specific") is False and len(f) == 1 and _has_harmonics(f[0], every):
            continue
        if f and all(not m.constant for m in f):
            hits.append(dict(lines=f, t0=float(np.mean([m.t0 for m in f])),
                             t1=float(np.mean([m.t1 for m in f]))))
    if not hits:
        return None
    hits.sort(key=lambda h: h["t0"])
    min_dur = float(entry.get("min_dur", DEFAULT_MIN_DUR))
    long_hits = [h for h in hits if h["t1"] - h["t0"] >= min_dur]
    pattern = entry.get("pattern", "any")
    cadence_ok = False
    if pattern == "continuous":
        if not long_hits:
            return None
        use = long_hits
    else:
        if len(hits) < 2 and not (pattern == "any" and long_hits):
            return None
        use = hits if len(hits) >= 2 else long_hits
        if "burst" in entry and len(hits) >= 3:
            on = float(np.median([h["t1"] - h["t0"] for h in hits]))
            off = float(np.median([b["t0"] - a["t1"] for a, b in zip(hits, hits[1:])]))
            cadence_ok = (abs(on - entry["burst"]) <= CADENCE_TOL * entry["burst"]
                          and abs(off - entry["gap"]) <= CADENCE_TOL * entry["gap"])
            if not cadence_ok and pattern == "cadenced":
                return None
    prom = float(np.mean([m.prom for h in use for m in h["lines"]]))
    if prom < MIN_PROM:
        return None
    return dict(entry=entry, hits=use, cadence_ok=cadence_ok, prom=prom)


def score(m: dict) -> float:
    e = m["entry"]
    s = 0.35 + (0.25 if len(e["freqs"]) > 1 else 0.0) + (0.2 if m["cadence_ok"] else 0.0)
    s += min(0.2, max(0.0, (m["prom"] - MIN_PROM) / 60))
    return min(s, 0.35 if not m["cadence_ok"] else 0.5) if e.get("specific") is False else min(s, 1.0)


def describe(m: dict) -> str:
    e, hits = m["entry"], m["hits"]
    fr = " and ".join(f"{np.mean([ln.f for h in hits for ln in h['lines'] if abs(ln.f - ef) <= _tol(ef)]):.0f}"
                      for ef in e["freqs"])
    when = ", ".join(f"{h['t0']:.0f} s" for h in hits[:4]) + (" ..." if len(hits) > 4 else "")
    s = (f"stable tone{'s' if len(e['freqs']) > 1 else ''} at {fr} Hz ({len(hits)} burst"
         f"{'s' if len(hits) != 1 else ''}, near {when}) are consistent with the **{e['name']}** "
         f"({e['where']})")
    if m["cadence_ok"]:
        s += "; on/off timing also fits"
    if e.get("specific") is False:
        s += "; a single frequency is shared by several systems, so this is weak evidence"
    return s


class SignalTonesProbe(Probe):
    name = "signal_tones"
    group = "Sounds"
    clutter_sensitive = True
    masked_by = ("sirens",)
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 600.0

    def run(self, path: Path) -> Finding:
        lines: list[Line] = []
        n = 0
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 1.5 or sr < 2000:
                    continue
                n += 1
                lines += [ln for ln in find_lines(x, sr, off) if ln.prom >= MIN_PROM]
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 1.5 s at 2 kHz or more")
        groups = group_simultaneous(dedupe(lines))
        matches = [m for e in _load("tones").get("tones", []) if (m := match_entry(e, groups))]
        if not matches:
            return Finding(
                probe=self.name, detected=False,
                summary="No stable tone signature matched the telephone or emergency-alert "
                        "tone table. Codecs, music and noise hide many tones.")
        matches.sort(key=score, reverse=True)
        matches = matches[:3]
        best = score(matches[0])
        regions = []
        for m in matches:
            for h in m["hits"][:3]:
                fs = [ln.f for ln in h["lines"]]
                regions.append(dict(label=f"{m['entry']['name']} near {h['t0']:.0f} s",
                                    t0=max(0.0, h["t0"] - 3), t1=h["t1"] + 3,
                                    f0=0.8 * min(fs), f1=1.2 * max(fs)))
        specific = any(m["entry"].get("specific") is not False for m in matches)
        return Finding(
            probe=self.name, detected=True, strength=best, narrows=0.3 if specific else 0.12,
            constrains=["region"], effort=2, regions=regions,
            follow_ups=follow_ups("audio", "tones"),
            summary="; ".join(describe(m) for m in matches)
            + ". Confirm the cadence and check that the audio is a live recording, not a "
              "replayed or synthesised tone.")
