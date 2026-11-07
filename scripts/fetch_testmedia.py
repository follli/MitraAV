"""Download the freely licensed test media listed in testmedia/manifest.json, politely.

One file at a time, a pause between files, Retry-After honoured on HTTP 429, size checked against the
manifest, and files already present are skipped. Wikimedia asks automated clients to be gentle, so
do not parallelise this or lower the pause.

    python scripts/fetch_testmedia.py            # fetch everything missing
    python scripts/fetch_testmedia.py --list     # show what is present or missing
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "testmedia"
UA = {"User-Agent": "MitraAV-test-media-fetch/0.1 (local development; one file at a time)"}
PAUSE = 8.0          # seconds between files
MAX_TRIES = 5


def fetch(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    wait = 30.0
    for attempt in range(1, MAX_TRIES + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise RuntimeError(f"size {tmp.stat().st_size} != expected {size}")
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == MAX_TRIES:
                raise
            wait = max(wait, float(e.headers.get("Retry-After", 0) or 0))
            print(f"    rate limited; waiting {wait:.0f} s (try {attempt}/{MAX_TRIES})")
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("gave up")


def main(argv: list[str]) -> int:
    items = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if "--list" in argv:
        for o in items:
            p = ROOT / o["dir"] / o["filename"]
            print(f"{'present' if p.exists() else 'MISSING':8s} {o['dir']}/{o['filename']}")
        return 0
    todo = [o for o in items if not (ROOT / o["dir"] / o["filename"]).exists()]
    print(f"{len(items) - len(todo)} present, {len(todo)} to fetch ({sum(o['size'] for o in todo) / 1e6:.1f} MB)")
    failed = 0
    for i, o in enumerate(todo):
        dest = ROOT / o["dir"] / o["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i + 1}/{len(todo)}] {o['dir']}/{o['filename']} ({o['size'] / 1e6:.2f} MB)")
        try:
            fetch(o["url"], dest, o["size"])
        except Exception as e:
            failed += 1
            print(f"    FAILED: {e}")
        if i < len(todo) - 1:
            time.sleep(PAUSE)
    print(f"done, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
