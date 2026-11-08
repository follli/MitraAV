"""Scene type: what kind of place or building the picture shows (street, church, desert, harbour...).

Uses the Places365 ResNet18 classifier (365 scene categories, 22.8 MB, runs offline on LiteRT; MIT
code, CC BY pretrained weights: Places Project, places2.csail.mit.edu). It reports what a person
could see in the picture, so it is shown quietly as context, never as a lead. Its job is to point
to the right checklists and references for that kind of scene (road furniture for a street,
architecture guides for a place of worship, climate zones for open country).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.models import SETUP_HINT, places_path
from ..core.probe import Probe
from ..core.references import follow_ups

MIN_P = 0.10          # candidates shown
MIN_TOP = 0.20        # below this nothing stands out
SURE = 0.50           # below this the guess is flagged as uncertain
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# category key -> label fragments (matched against Places365 names such as "church/outdoor")
KINDS = {
    "street": ("street", "highway", "crosswalk", "alley", "driveway", "parking_lot", "gas_station",
               "bridge", "overpass", "downtown", "plaza", "residential_neighborhood", "storefront",
               "market/outdoor", "bazaar/outdoor", "toll", "boardwalk", "airfield", "runway",
               "train_station", "bus_station", "subway_station", "railroad_track", "desert_road",
               "highway"),
    "worship": ("church", "cathedral", "mosque", "temple", "synagogue", "pagoda", "shrine", "chapel"),
    "landmark": ("castle", "palace", "tower", "lighthouse", "skyscraper", "office_building",
                 "apartment_building", "barn", "windmill", "stadium", "museum", "ruin", "amphitheater",
                 "arch", "fountain", "monument", "campus", "embassy", "courthouse", "greenhouse",
                 "building_facade", "hotel/outdoor", "pavilion", "house", "village"),
    "nature": ("desert", "forest", "rainforest", "jungle", "rice_paddy", "field", "mountain",
               "glacier", "snowfield", "beach", "coast", "ocean", "harbor", "lake", "river",
               "canyon", "tundra", "swamp", "marsh", "valley", "volcano", "cliff", "farm", "orchard",
               "vineyard", "pasture", "hayfield", "badlands", "dune", "iceberg", "waterfall",
               "bamboo", "park", "garden", "hot_spring", "oasis", "ski_slope", "islet", "pond"),
}
READABLE_HEAD = {"worship": "place of worship", "landmark": "notable building or structure",
                 "street": "street, road or transport scene", "nature": "natural or open landscape"}


def load_labels() -> tuple[list[str], list[str]]:
    """(category names like 'church/outdoor', 'indoor'|'outdoor' per category)."""
    from importlib import resources
    ref = resources.files("mitraav") / "references"
    names = [ln.split()[0][3:] for ln in (ref / "places365_categories.txt").read_text().splitlines() if ln.strip()]
    io = [("indoor" if ln.split()[1] == "1" else "outdoor")
          for ln in (ref / "places365_io.txt").read_text().splitlines() if ln.strip()]
    return names, io


def kind_of(name: str) -> str | None:
    for kind, frags in KINDS.items():
        if any(f in name for f in frags):
            return kind
    return None


def readable(name: str) -> str:
    base, _, sub = name.partition("/")
    base = base.replace("_", " ")
    return f"{base} ({sub})" if sub else base


def _views(img: Image.Image) -> list[np.ndarray]:
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    big = img.resize((max(224, round(w * s)), max(224, round(h * s))))
    left, top = (big.width - 224) // 2, (big.height - 224) // 2
    crop = big.crop((left, top, left + 224, top + 224))
    squash = img.resize((224, 224))
    out = []
    for v in (crop, squash):
        x = (np.asarray(v, np.float32) / 255 - MEAN) / STD
        out.append(x.transpose(2, 0, 1)[None].astype(np.float32))
    return out


def classify(path: Path, model_path: Path) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    probs = []
    with Image.open(path) as img:
        views = _views(img)
    for x in views:
        it.set_tensor(inp["index"], x)
        it.invoke()
        z = it.get_tensor(out["index"])[0].astype(np.float64)
        e = np.exp(z - z.max())
        probs.append(e / e.sum())
    return np.mean(probs, axis=0)


class SceneTypeProbe(Probe):
    name = "scene_type"
    group = "Visual context"
    quiet = True
    order = 10               # early: later picture probes read the scene from the shared context
    media = (MediaType.IMAGE,)
    timeout = 90.0

    def run(self, path: Path) -> Finding:
        if importlib.util.find_spec("ai_edge_litert") is None:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Scene recognition is an add-on: install it from Add-ons in the app, or "
                                  "run: mitraav-setup --install scene")
        mp = places_path()
        if not mp.exists():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"Scene model not downloaded (scans never download): {SETUP_HINT}")
        p = classify(path, mp)
        names, io = load_labels()
        order = [i for i in np.argsort(-p) if p[i] >= MIN_P][:3]
        if not order or p[order[0]] < MIN_TOP:
            return Finding(probe=self.name, detected=False,
                           summary="No scene category stood out; the picture may be unusual, "
                                   "close-up or heavily edited.")
        unsure = p[order[0]] < SURE
        parts = [f"**{readable(names[i])}** ({p[i] * 100:.0f}%)" for i in order]
        kinds = []
        for i in order:
            k = kind_of(names[i])
            if k and k not in kinds:
                kinds.append(k)
        setting = io[order[0]]
        fu = []
        for k in kinds:
            fu += follow_ups("visual", "reverse" if k == "landmark" else k)
        notes = [(f"the scene model is unsure; it may be {', '.join(parts)}, {setting}. A person or "
                  "object in the foreground often confuses it, so check the picture yourself"
                  if unsure else f"the scene looks like {', '.join(parts)}, {setting}")]
        for k in kinds:
            notes.append({
                "street": "for a street or road scene, compare road furniture with the references "
                          "(bollards, utility poles, road markings, plate formats, driving side)",
                "worship": "for a place of worship, compare the architecture with the references "
                           "and try a reverse image search",
                "landmark": "for a building or structure, a reverse image search is the quickest "
                            "way to a name",
                "nature": "for open country, compare the terrain and vegetation with climate "
                          "zone and biome maps"}[k])
        return Finding(
            probe=self.name, detected=True, strength=float(p[order[0]]), narrows=0.05, pill="context",
            meta={"context": {"scene": dict(path=str(path), kind=kinds[0] if kinds else None,
                                            label=readable(names[order[0]]), p=float(p[order[0]]),
                                            kinds=kinds)}},
            constrains=["setting"], effort=1, follow_ups=fu,
            summary="; ".join(notes) + ". This is what the picture shows to the eye, so treat it "
            "as context; the scene model is coarse and can be wrong.")
