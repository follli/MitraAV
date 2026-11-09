"""Coarse location hints and how they combine.

Probes publish what they imply about *where* in `Finding.meta["geo"]` as
`{"label": "...", "any_of": [macro-region ids], "weight": 0..1}`. `combine` then reports which
macro-regions the hints share and which hints disagree. Everything is deliberately coarse (seven
macro-regions, with neighbours included where a border is blurry) and every disagreement is worded as a
question, because there are many innocent reasons for one: foreign signage and phone numbers, a
migratory or widely ranging species, a re-uploaded or edited file.
"""
from __future__ import annotations

MACROS = {
    "europe": "Europe",
    "north_america": "North America",
    "central_america": "Central America and the Caribbean",
    "south_america": "South America",
    "africa": "Africa",
    "asia": "Asia",
    "oceania": "Australia, New Zealand and the Pacific",
}
EU, NA, CA, SA, AF, AS, OC = ("europe", "north_america", "central_america", "south_america",
                              "africa", "asia", "oceania")

# BirdNET region names (core/birdgeo.region_name) -> macro-regions (neighbours included)
BIRD_REGIONS = {
    "Central America and the Caribbean": {CA, NA, SA},
    "eastern North America": {NA},
    "western North America": {NA},
    "South America": {SA, CA},
    "Europe": {EU},
    "Africa": {AF},
    "the Middle East and central Asia": {AS},
    "Siberia and northern Asia": {AS, EU},
    "East Asia": {AS},
    "Australia and New Zealand": {OC},
    "Southeast Asia and the Pacific islands": {AS, OC},
}

# country (as written in references/codes.yaml) -> macro-regions
COUNTRIES = {
    "Argentina": {SA}, "Australia": {OC}, "Austria": {EU}, "Belgium": {EU}, "Brazil": {SA},
    "Canada": {NA}, "Chile": {SA}, "China": {AS}, "Colombia": {SA}, "Cuba": {CA},
    "Denmark": {EU}, "Egypt": {AF}, "Finland": {EU}, "France": {EU}, "Germany": {EU},
    "Greece": {EU}, "Hungary": {EU}, "India": {AS}, "Indonesia": {AS, OC}, "Iran": {AS},
    "Ireland": {EU}, "Israel": {AS}, "Italy": {EU}, "Japan": {AS}, "Kazakhstan": {AS},
    "Kenya": {AF}, "Malaysia": {AS}, "Mexico": {NA, CA}, "Morocco": {AF}, "New Zealand": {OC},
    "Nigeria": {AF}, "Norway": {EU}, "Pakistan": {AS}, "Peru": {SA}, "Poland": {EU},
    "Portugal": {EU}, "Romania": {EU}, "Russia": {EU, AS}, "Singapore": {AS},
    "South Africa": {AF}, "South Korea": {AS}, "Spain": {EU}, "Sri Lanka": {AS},
    "Sweden": {EU}, "Switzerland": {EU}, "Thailand": {AS}, "Turkey": {EU, AS}, "Ukraine": {EU},
    "Venezuela": {SA}, "Vietnam": {AS}, "the Czech Republic": {EU}, "the Netherlands": {EU},
    "the Philippines": {AS, OC}, "the United Arab Emirates": {AS}, "the United Kingdom": {EU},
    "the euro area": {EU}, "the US": {NA}, "the UK": {EU}, "another NANP country": {NA, CA},
}
_ORDERED = sorted(COUNTRIES, key=len, reverse=True)         # longest name first


def macros_for_phrase(text: str) -> set[str]:
    """Macro-regions named in a phrase such as 'Japan or China (yen or yuan)' (empty if none)."""
    out: set[str] = set()
    rest = text
    for name in _ORDERED:
        if name in rest:
            out |= COUNTRIES[name]
            rest = rest.replace(name, " ")
    return out


def macros_for_latlon(lat: float, lon: float) -> set[str]:
    from .birdgeo import region_name
    return set(BIRD_REGIONS.get(region_name(lat, lon), set()))


def hint(label: str, any_of: set[str], weight: float) -> dict | None:
    """The dict a probe stores in meta["geo"] (None when there is nothing to say)."""
    if not any_of:
        return None
    return dict(label=label, any_of=sorted(any_of), weight=weight)


def combine(hints: list[tuple[str, dict]]) -> dict:
    """hints: (probe name, geo dict). Returns the support per macro-region and the disagreements."""
    support: dict[str, float] = {}
    for _, h in hints:
        for m in h["any_of"]:
            support[m] = support.get(m, 0.0) + h["weight"] / len(h["any_of"]) ** 0.5
    conflicts = []
    for i, (pa, a) in enumerate(hints):
        for pb, b in hints[i + 1:]:
            if pa != pb and not set(a["any_of"]) & set(b["any_of"]) and min(a["weight"], b["weight"]) >= 0.2:
                conflicts.append((pa, a, pb, b))
    common = set(MACROS)
    for _, h in hints:
        common &= set(h["any_of"])
    return dict(support=dict(sorted(support.items(), key=lambda kv: -kv[1])), common=sorted(common),
                conflicts=conflicts, n=len(hints))
