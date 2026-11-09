"""Scan reports (Markdown and JSON) and the location summary they open with."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .finding import Finding, plain
from .geo import MACROS, combine


def geo_hints(findings: list[Finding]) -> list[tuple[str, dict]]:
    return [(f.probe, f.meta["geo"]) for f in findings
            if f.status == "ok" and not f.inconclusive and f.meta.get("geo")]


def _names(macros: list[str]) -> str:
    return " or ".join(MACROS[m] for m in macros)


def location_summary(findings: list[Finding], display=lambda n: n.replace("_", " ")) -> dict:
    """What the probes' coarse location hints say together: support per macro-region, the region(s)
    every hint allows, and pairs of hints that disagree (as questions, not accusations)."""
    hints = geo_hints(findings)
    c = combine(hints)
    lines = []
    for probe, h in hints:
        lines.append(f"{display(probe)}: {h['label']} is consistent with {_names(h['any_of'])}")
    notes = []
    if c["n"] >= 2 and c["common"]:
        notes.append(f"All {c['n']} hints are consistent with **{_names(c['common'])}**.")
    elif c["n"] == 1:
        notes.append("Only one probe offers a location hint, so nothing can be cross-checked yet.")
    for pa, a, pb, b in c["conflicts"]:
        notes.append(
            f"**Possible disagreement:** {display(pa)} ({a['label']}) points to {_names(a['any_of'])}, "
            f"but {display(pb)} ({b['label']}) points to {_names(b['any_of'])}. Innocent explanations "
            "include foreign signage or phone numbers, a widely ranging or migratory species, a "
            "re-uploaded file, or edited metadata; check which clue is the more reliable.")
    grids = {}
    for f in findings:
        if f.status == "ok" and not f.inconclusive and f.meta.get("grid"):
            grids.setdefault(f.meta["grid"]["hz"], []).append(display(f.probe))
    if len(grids) > 1:
        notes.append("**Possible disagreement:** the mains frequency is " + " but ".join(
            f"{hz} Hz according to {', '.join(names)}" for hz, names in sorted(grids.items()))
            + ". Both cannot be right for one place and time: check for a recording that was combined, converted "
              "(a speed change shifts hum) or filmed under screens or LED drivers.")
    return dict(hints=lines, notes=notes, support=c["support"], common=c["common"],
                conflicts=[(display(pa), display(pb)) for pa, _, pb, _ in c["conflicts"]], n=c["n"])


def _classify(findings: list[Finding]) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {"leads": [], "inconclusive": [], "nothing": [], "not_run": [], "context": []}
    for f in sorted(findings, key=lambda f: (-f.score, f.probe)):
        if f.status != "ok":
            out["not_run"].append(f)
        elif f.inconclusive:
            out["inconclusive"].append(f)
        elif f.detected and f.quiet:
            out["context"].append(f)
        elif f.detected:
            out["leads"].append(f)
        else:
            out["nothing"].append(f)
    return out


def to_json(path: Path, findings: list[Finding]) -> str:
    return json.dumps(dict(file=path.name, generated=datetime.now().isoformat(timespec="seconds"),
                           location=location_summary(findings),
                           findings=[f.to_dict() for f in sorted(findings, key=lambda f: -f.score)]),
                      indent=2, default=str)


def to_markdown(path: Path, findings: list[Finding], display=lambda n: n.replace("_", " ")) -> str:
    g = _classify(findings)
    loc = location_summary(findings, display)
    out = [f"# MitraAV report: {path.name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. {len(g['leads'])} lead(s) from {len(findings)} probe(s). "
           "Everything below is a lead, not a conclusion.", ""]
    out += ["## Where this points", ""]
    if loc["hints"]:
        out += [f"- {plain(h)}" for h in loc["hints"]] + [""]
        out += [f"{plain(n)}" for n in loc["notes"]] + [""]
    else:
        out += ["No probe offered a location hint.", ""]

    def block(f: Finding) -> list[str]:
        text = plain(f.summary) or f.detail
        lines = [f"### {display(f.probe)}", "", text, ""]
        if f.regions:
            lines.append("Where in the recording: " + "; ".join(r["label"] for r in f.regions[:6]) + ".\n")
        if f.follow_ups:
            lines += ["Where to look next:"] + [f"- [{u.name}]({u.url})" + (f": {u.note}" if u.note else "")
                                                if u.url else f"- {u.name}" for u in f.follow_ups] + [""]
        return lines

    if g["leads"]:
        out += ["## Top leads", ""]
        for f in g["leads"][:3]:
            out += block(f)
        if g["leads"][3:]:
            out += ["## Other leads", ""]
            for f in g["leads"][3:]:
                out += block(f)
    else:
        out += ["## Leads", "", "None.", ""]
    if g["inconclusive"]:
        out += ["## Inconclusive", ""] + [f"- **{display(f.probe)}:** {plain(f.caveat)}" for f in g["inconclusive"]] + [""]
    if g["nothing"]:
        out += ["## Checked, nothing found", ""] + [f"- {display(f.probe)}" for f in g["nothing"]] + [""]
    if g["not_run"]:
        out += ["## Could not run", ""] + [f"- {display(f.probe)}: {f.detail}" for f in g["not_run"]] + [""]
    if g["context"]:
        out += ["## Context (what the picture shows to the eye)", ""]
        out += [f"- **{display(f.probe)}:** {plain(f.summary)}" for f in g["context"]] + [""]
    return "\n".join(out).rstrip() + "\n"
