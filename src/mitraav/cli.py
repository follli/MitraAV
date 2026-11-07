from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .core.finding import plain
from .core.runner import scan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mitraav-scan", description="Cheap first-pass lead scan")
    ap.add_argument("--version", action="version", version=f"mitraav {__version__}")
    ap.add_argument("file", type=Path)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if not args.file.is_file():
        ap.error(f"not a file: {args.file}")
    findings = scan(args.file)

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
        return 0
    if not findings:
        print("No probes apply to this file type yet.")
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
