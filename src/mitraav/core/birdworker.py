"""Runs BirdNET in its own process (it spawns workers, which a daemonic probe process may not).

Usage: python -m mitraav.core.birdworker FILE   -> JSON list of detections on stdout.
Never downloads the model: the probe checks for it first (see model_path()).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL = ("birdnet", "acoustic-models", "v2.4", "tf", "model-fp32.tflite")
MIN_CONF = 0.3
REPORT_CONF = 0.5
GEO = ("birdnet", "geo-models", "v2.4", "tf", "model-fp32.tflite")


# BirdNET also labels non-bird sounds. They are never species; human voices mean busy audio.
PSEUDO = {"Dog", "Engine", "Environmental", "Fireworks", "Gun", "Human non-vocal", "Human vocal",
          "Human whistle", "Noise", "Power tools", "Siren"}


def human_segments(dets: list[dict]) -> int:
    """Distinct 3 s segments where BirdNET heard a human voice (>= 0.5)."""
    return len({round(d["t0"] / 3) for d in dets if d["sci"] == "Human vocal" and d["conf"] >= 0.5})


def summarise(dets: list[dict]) -> list[dict]:
    """Per species: segments, best confidence and time span; keep credible ones."""
    from collections import defaultdict
    seen = set()
    by = defaultdict(list)
    for d in sorted(dets, key=lambda d: d["t0"]):
        if d["sci"] in PSEUDO or d["sci"] == d["common"]:
            continue                                      # not a species
        key = (d["sci"], round(d["t0"] / 3))          # overlapping chunks repeat segments
        if key in seen:
            continue
        seen.add(key)
        by[d["sci"]].append(d)
    out = []
    for sci, ds in by.items():
        good = [d for d in ds if d["conf"] >= REPORT_CONF]
        best = max(d["conf"] for d in ds)
        if not good or (len(good) < 2 and best < 0.9):
            continue
        out.append(dict(sci=sci, common=ds[0]["common"], n=len(good), best=best,
                        t0=min(d["t0"] for d in good), t1=max(d["t1"] for d in good)))
    return sorted(out, key=lambda s: (-s["n"] * s["best"]))


def geo_model_path() -> Path | None:
    mp = model_path()
    return None if mp is None else mp.parents[4].joinpath(*GEO)


def model_path() -> Path | None:
    """Where BirdNET keeps the acoustic model, or None if birdnet is not installed."""
    try:
        from birdnet.utils.local_data import get_app_data_path
    except Exception:
        return None
    return Path(get_app_data_path()).joinpath(*MODEL)


def main(path: str) -> None:
    import birdnet
    import numpy as np

    from mitraav.core.audio import iter_chunks

    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    out = []
    for off, x, sr in iter_chunks(Path(path)):
        r = model.predict_arrays((x.astype(np.float32), sr), top_k=3, overlap_duration_s=1.0,
                                 default_confidence_threshold=MIN_CONF)
        df = r.to_dataframe()
        for row in df.itertuples():
            sci, _, common = str(row.species_name).partition("_")
            out.append(dict(t0=off + float(row.start_time), t1=off + float(row.end_time),
                            sci=sci, common=common, conf=float(row.confidence)))
    geo = None
    gp = geo_model_path()
    sp = summarise(out)
    if sp and gp is not None and gp.exists():
        from mitraav.core.birdgeo import locate
        g = birdnet.load("geo", "2.4", "tf", library="litert")
        names = [str(n) for n in g.species_list]
        idx, w = [], []
        for s_ in sp[:8]:
            hit = next((i for i, n in enumerate(names) if n.startswith(s_["sci"] + "_")), None)
            if hit is not None:
                idx.append(hit)
                w.append(s_["best"] * float(np.log1p(s_["n"])))
        geo = locate(lambda la, lo, wk: np.asarray(
            g.predict(la, lo, week=wk, min_confidence=0.0).species_probs), idx, w)
    sys.stdout.write("\n@@JSON@@" + json.dumps(dict(dets=out, geo=geo)))


if __name__ == "__main__":
    main(sys.argv[1])
