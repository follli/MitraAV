"""Where could this set of species co-occur? Scores a world grid with BirdNET's geo model.

The geo model returns, for a latitude, longitude and week, the probability that each species is
present. Per cell we sum weight * log(probability) over the detected species, floored so one
mistaken species cannot veto a region. Stage 1 uses the year-round maximum (an upper bound for
every week), stage 2 re-scores the best cells week by week. The result is a ranked set of regions
and seasons, a lead for the region and season constraints, never a location.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

SATURATE = 0.5      # a species at or above this probability counts fully; none counts as zero
STEP = 3.0
TOP_CELLS = 300
CLUSTER_KM = 800.0
WEEKS = range(1, 49, 2)    # BirdNET weeks (4 per month), every second one

Predict = Callable[[float, float, int | None], np.ndarray]


def weeks_to_text(weeks: list[int]) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if not weeks:
        return "year-round"
    ws = sorted(set(weeks))
    runs, cur = [], [ws[0]]
    for w in ws[1:]:
        if w - cur[-1] <= 2:
            cur.append(w)
        else:
            runs.append(cur)
            cur = [w]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 1 and runs[-1][-1] >= 47:     # wraps around new year
        runs = [runs[-1] + runs[0]] + runs[1:-1]
    r = max(runs, key=len)
    a, b = r[0], r[-1]
    return f"{months[(a - 1) // 4]} to {months[(b - 1) // 4]}" if (a - 1) // 4 != (b - 1) // 4 \
        else months[(a - 1) // 4]


def region_name(lat: float, lon: float) -> str:
    """A coarse, deliberately general name (boxes include nearby coastal waters)."""
    def inside(la0, la1, lo0, lo1):
        return la0 <= lat <= la1 and lo0 <= lon <= lo1
    if inside(5, 24, -120, -58):
        return "Central America and the Caribbean"
    if inside(24, 62, -100, -52):
        return "eastern North America"
    if inside(24, 78, -170, -100):
        return "western North America"
    if inside(-60, 5, -85, -30):
        return "South America"
    if inside(46, 72, -12, 60) or inside(35, 46, -12, 28):     # includes European Russia to the Urals
        return "Europe"
    if inside(-36, 35, -20, 52):
        return "Africa"
    if inside(12, 55, 28, 90):
        return "the Middle East and central Asia"
    if inside(55, 78, 60, 180):
        return "Siberia and northern Asia"
    if inside(15, 60, 90, 150):
        return "East Asia"
    if inside(-48, -10, 110, 180):
        return "Australia and New Zealand"
    if inside(-10, 15, 90, 180):
        return "Southeast Asia and the Pacific islands"
    return "open ocean or other regions"


def merge_regions(clusters: list[dict], top_score: float | None = None) -> list[dict]:
    """One entry per named region, by its best cluster. `fit` is how well it matches compared with
    the best region (1.0 = the best; 0.37 or more = nearly as well)."""
    top = top_score if top_score is not None else max((c["best_score"] for c in clusters), default=0.0)
    out: dict[str, dict] = {}
    for c in clusters:
        r = out.setdefault(c["region"], dict(region=c["region"], share=0.0, seasons=[],
                                             fit=0.0))
        r["share"] += c["share"]
        r["fit"] = max(r["fit"], float(np.exp(c.get("best_score", top) - top)))
        if c["season"] and c["season"] not in r["seasons"]:
            r["seasons"].append(c["season"])
    return sorted(out.values(), key=lambda r: (-r["fit"], -r["share"]))


def _km(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 111.0,
                    (lon1 - lon2) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2)))


def locate(predict: Predict, species_idx: list[int], weights: list[float]) -> dict | None:
    """predict(lat, lon, week|None) -> probabilities for all species; species_idx picks ours."""
    if not species_idx:
        return None
    w = np.asarray(weights, dtype=float)
    lats = np.arange(-60.0, 80.1, STEP)
    lons = np.arange(-180.0, 180.0, STEP)

    def score(p: np.ndarray) -> float:
        # saturating, not logarithmic: a species the model puts nowhere here (often a mistaken ID)
        # costs its weight but cannot veto a region the other species agree on
        return float((w * np.minimum(p[species_idx], SATURATE) / SATURATE).sum())

    s1 = np.array([[score(predict(float(la), float(lo), None)) for lo in lons] for la in lats])
    flat = np.argsort(-s1.ravel())[:TOP_CELLS]
    cells = []
    for f in flat:
        a, b = np.unravel_index(f, s1.shape)
        la, lo = float(lats[a]), float(lons[b])
        ws = np.array([score(predict(la, lo, wk)) for wk in WEEKS])
        best = float(ws.max())
        season = [wk for wk, v in zip(WEEKS, ws) if v >= best - 0.5]
        cells.append(dict(lat=la, lon=lo, score=best, weeks=season))
    cells.sort(key=lambda c: -c["score"])
    top = cells[0]["score"]
    mass = np.array([np.exp(c["score"] - top) for c in cells])        # score is in weight units
    total = float(mass.sum())
    left = list(range(len(cells)))
    clusters = []
    while left and len(clusters) < 3:
        c0 = cells[left[0]]
        mem = [i for i in left if _km(c0["lat"], c0["lon"], cells[i]["lat"], cells[i]["lon"]) <= CLUSTER_KM]
        m = mass[mem]
        wk = [wk for i in mem for wk in cells[i]["weeks"]]
        counts = np.bincount(wk, minlength=49)
        common = [k for k in WEEKS if counts[k] >= 0.5 * counts.max()]
        if len(common) >= 21:                                  # most of the year
            common = []
        clusters.append(dict(
            lat=float(np.average([cells[i]["lat"] for i in mem], weights=m)),
            lon=float(np.average([cells[i]["lon"] for i in mem], weights=m)),
            best_score=float(cells[mem[0]]["score"]), share=float(m.sum() / total),
            season=weeks_to_text(common), n_cells=len(mem)))
        clusters[-1]["region"] = region_name(clusters[-1]["lat"], clusters[-1]["lon"])
        left = [i for i in left if i not in mem]
    return dict(clusters=clusters, regions=merge_regions(clusters, top), top_score=top)
