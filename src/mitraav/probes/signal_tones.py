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


NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MEASURE_PROM = 18.0      # unmatched tones must stand well clear of the floor
MEASURE_RANGE_DB = 30.0  # ... and be within this many dB of the strongest tone
MAX_TONES = 6            # more distinct pitches than this is music, speech or noise, not a signal


def note_name(f: float) -> str:
    m = int(round(69 + 12 * np.log2(f / 440.0)))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def _fundamentals(lines: list[Line]) -> list[tuple[Line, int]]:
    """Lines that are not an overtone (2x..6x) of a lower simultaneous line, with how many overtones
    each has: a struck chime or beeper is one pitch with a harmonic stack, not several tones."""
    out = []
    for ln in lines:
        is_partial = False
        for m in lines:
            if m.f >= ln.f:
                continue
            ov = min(m.t1, ln.t1) - max(m.t0, ln.t0)
            if ov >= 0.5 * min(ln.t1 - ln.t0, m.t1 - m.t0) and any(abs(ln.f - k * m.f) <= 0.015 * ln.f for k in range(2, 7)):
                is_partial = True
                break
        if not is_partial:
            n = sum(1 for o in lines if o.f > ln.f and min(o.t1, ln.t1) - max(o.t0, ln.t0) >= 0.5 * min(o.t1 - o.t0, ln.t1 - ln.t0)
                    and any(abs(o.f - k * ln.f) <= 0.015 * o.f for k in range(2, 7)))
            out.append((ln, n))
    return out


def measure_tones(lines: list[Line]) -> list[dict]:
    """Stable tones that match no table entry: one entry per pitch (within 1.5%), overtones folded
    into their fundamental, with how often it sounded and its on/off timing. Empty unless there is
    evidence of a deliberate signal (repeated, or a structured tone with overtones) and not so many
    pitches that it is music, speech or noise."""
    cands = [ln for ln in lines if not ln.constant]
    funds = [(ln, n) for ln, n in _fundamentals(cands) if ln.prom >= MEASURE_PROM]
    if not funds:
        return []
    top = max(ln.level for ln, _ in funds)
    funds = [(ln, n) for ln, n in funds if ln.level >= top - MEASURE_RANGE_DB]
    pitches: list[dict] = []
    for ln, n in sorted(funds, key=lambda x: x[0].f):
        if pitches and abs(ln.f - pitches[-1]["f"]) <= 0.015 * ln.f:
            c = pitches[-1]
            c["lines"].append(ln)
            c["partials"] = max(c["partials"], n)
            c["f"] = float(np.mean([m.f for m in c["lines"]]))
        else:
            pitches.append(dict(f=ln.f, lines=[ln], partials=n))
    if len(pitches) > MAX_TONES:
        return []
    bursts = sum(len(c["lines"]) for c in pitches)
    if bursts < 2 and not any(c["partials"] for c in pitches):
        return []                                   # one bare tone: too easily a whine or whistle
    out = []
    for c in pitches:
        ls = sorted(c["lines"], key=lambda m: m.t0)
        on = float(np.median([m.t1 - m.t0 for m in ls]))
        off = float(np.median([b.t0 - a.t1 for a, b in zip(ls, ls[1:])])) if len(ls) > 1 else None
        out.append(dict(f=c["f"], n=len(ls), t0=ls[0].t0, t1=ls[-1].t1, on=on, off=off,
                        partials=c["partials"], level=max(m.level for m in ls)))
    return sorted(out, key=lambda c: c["t0"])                 # in order of sounding: a chime's pitch sequence


def measured_finding(name: str, tones: list[dict]) -> Finding:
    parts = []
    for c in tones[:4]:
        pulse = ""
        if c["n"] > 1 and c["off"] is not None:
            pulse = f", {c['n']} bursts of about {c['on']:.2f} s with about {max(c['off'], 0):.2f} s gaps"
        elif c["n"] == 1:
            pulse = f", one tone of about {c['on']:.1f} s"
        parts.append(f"**{c['f']:.0f} Hz ({note_name(c['f'])})** near {c['t0']:.0f} s{pulse}")
    strength = min(0.45, 0.2 + 0.05 * min(sum(c["n"] for c in tones), 5))
    regions = [dict(label=f"tone {c['f']:.0f} Hz near {c['t0']:.0f} s", t0=max(0.0, c["t0"] - 2),
                    t1=c["t1"] + 2, f0=0.8 * c["f"], f1=1.2 * c["f"]) for c in tones[:3]]
    return Finding(
        probe=name, detected=True, strength=strength, narrows=0.1, constrains=["source type"], effort=2,
        regions=regions, follow_ups=follow_ups("audio", "tones"),
        summary="stable tone" + ("s" if len(tones) > 1 else "") + " measured: " + "; ".join(parts)
        + ". Not in the telephone or emergency-alert table, so this is a measured pitch pattern, "
          "consistent with a beeper, chime or alert (lift, door, appliance, vehicle). Compare the "
          "pitch and rhythm with recordings of the suspected device.")


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
            measured = measure_tones(dedupe(lines))
            if measured:
                return measured_finding(self.name, measured)
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
