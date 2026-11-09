from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .core.finding import plain
from .core.report import location_summary, to_json, to_markdown
from .core.runner import scan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mitraav-scan", description="Cheap first-pass lead scan")
    ap.add_argument("--version", action="version", version=f"mitraav {__version__}")
    ap.add_argument("file", type=Path, help="a file to scan, or a folder with --duplicates")
    ap.add_argument("--duplicates", action="store_true",
                    help="with a folder: list near-duplicate pictures, videos and recordings instead of scanning")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--report", type=Path, metavar="FILE",
                    help="also save a report (.md Markdown, .json JSON)")
    args = ap.parse_args(argv)

    if args.duplicates:
        if not args.file.is_dir():
            ap.error("--duplicates needs a folder")
        from .core.dupes import find_groups
        groups = find_groups(args.file)
        if args.json:
            print(json.dumps([[str(p) for p in g] for g in groups], indent=2))
        elif not groups:
            print("No near-duplicates found.")
        for i, g in enumerate(groups, 1):
            if not args.json:
                print(f"Group {i}:")
                for p in g:
                    print(f"  {p}")
        return 0
    if not args.file.is_file():
        ap.error(f"not a file: {args.file}")
    findings = scan(args.file)

    if args.report:
        args.report.write_text(to_json(args.file, findings) if args.report.suffix.lower() == ".json"
                               else to_markdown(args.file, findings), encoding="utf-8")
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
        return 0
    if not findings:
        print("No probes apply to this file type yet.")
    loc = location_summary(findings)
    for h in loc["hints"]:
        print(f"[@] {plain(h)}")
    for n in loc["notes"]:
        print(f"[@] {plain(n)}")
    for f in findings:
        tag = f"{f.score:.2f}" if f.status == "ok" else f.status.upper()
        mark = "+" if f.detected else "-"
        print(f"[{mark}] {f.probe:<12} {tag:>7}  {plain(f.summary) or f.detail}")
        if f.caveat:
            print(f"      ! inconclusive: {f.caveat}")
        for u in f.follow_ups:
            print(f"      -> {u.name} {u.url}  {u.note}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
