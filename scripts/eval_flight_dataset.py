"""Score the Doppler probe against the Zenodo aeroplane dataset (smartphone recordings of 40 real passes
with ADS-B tracks; folder layout raw_flight_data/V_*_H_*/<case>/).

For each case the authors' hand-picked frequency ridge (frequency_vs_time_used.csv) is fitted with the
probe's own Doppler model to get a reference closest-approach time, speed and slant distance. A case counts
as FOUND when some event reported by the probe has its closest approach within TOL seconds of the
reference. Folders named V_varying / H_varying break the model's straight, level, constant-speed
assumptions, so only the first folder is a fair test of the fit itself.

    python scripts/eval_flight_dataset.py testmedia/raw_flight_data/raw_flight_data [folder ...]
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from mitraav.probes import doppler as D

TOL = 5.0


def reference(case: Path, wav: Path) -> dict | None:
    rows = list(csv.DictReader(open(case / "frequency_vs_time_used.csv")))
    stamp = wav.stem.split("_", 2)[2]                          # 2026-05-13T20-03-42.021Z
    start = datetime.fromisoformat(stamp[:10] + "T" + stamp.split("T")[1][:8].replace("-", ":")
                                   + "." + stamp.split(".")[1].replace("Z", "") + "+00:00")
    t = np.array([(datetime.fromisoformat(r["time_utc"].replace("Z", "+00:00")) - start).total_seconds() for r in rows])
    f = np.array([float(r["frequency_hz"]) for r in rows])
    fit = D.fit_doppler(t, f)
    if not fit:
        return None
    p, r2 = fit
    return dict(t0=float(p[2]), v=float(p[1] * D.C), dist=float(p[1] * D.C * p[3]), r2=float(r2))


def events(summary: str) -> list[dict]:
    out = []
    for m in re.finditer(r"event (\d+)\*\*: closest approach heard near (\d+) s.*?(?:about \*\*(\d+) m/s\*\*.*?distance about "
                         r"\*\*(\d+) m\*\*|speed at least \*\*(\d+) m/s\*\*)", summary):
        out.append(dict(t0=float(m[2]), v=float(m[3] or m[5]), dist=float(m[4]) if m[4] else None, lower_bound=m[3] is None))
    return out


def main() -> int:
    base = Path(sys.argv[1])
    folders = sys.argv[2:] or sorted(p.name for p in base.iterdir() if p.is_dir())
    found = total = 0
    for folder in folders:
        print(f"== {folder}")
        for case in sorted((base / folder).iterdir()):
            wav = next(case.glob("*.wav"), None)
            ref = reference(case, wav) if wav else None
            if not ref:
                continue
            res = D.DopplerProbe().run(wav)
            evs = events(res.summary) if res.detected else []
            hit = next((e for e in evs if abs(e["t0"] - ref["t0"]) <= TOL), None)
            total += 1
            found += bool(hit)
            ours = (f"t0 {hit['t0']:.0f}s " + (f"v >= {hit['v']:.0f}" if hit["lower_bound"] else f"v {hit['v']:.0f} d {hit['dist']:.0f}")
                    if hit else ("flagged other events only" if evs else "nothing found"))
            print(f"{case.name:13s} ref t0 {ref['t0']:4.0f}s v {ref['v']:4.0f} m/s d {ref['dist']:5.0f} m | "
                  f"{'FOUND ' if hit else 'MISSED'} {ours}")
    print(f"\nfound {found} of {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
