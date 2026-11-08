"""Living things in a picture: plant, animal and fungus species candidates (BioCLIP).

Optional and heavy (see core/nature.py). The whole picture and, separately, its four quarters and
centre are classified. BioCLIP names a species for almost any picture, so the scene model first marks
building and street scenes: a building scene is not trusted as a whole, but a region that is not
itself clearly a building can still yield a species (a tree beside a cathedral), at a higher
threshold. A species is a lead to its range (GBIF, iNaturalist), which says where it grows, not where
the photo was taken. People are never reported.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

from PIL import Image

from ..core.finding import Finding, FollowUp, MediaType
from ..core.nature import INSTALL, MODEL_STR, files_present, installed
from ..core.probe import Probe

MIN_SPECIES = 0.50
MIN_GENUS = 0.60
SKIP = ("homo sapiens", "homo")


def _kingdom_word(k: str) -> str:
    return {"Plantae": "plant", "Animalia": "animal", "Fungi": "fungus"}.get(k, "organism")


BUILT = ("street", "worship", "landmark")      # scene kinds where a species guess is usually a hallucination
BUILT_P = 0.30                                  # whole picture: how sure the scene model must be
TILE_BUILT_P = 0.40                             # a region this clearly a building is skipped
MIN_TILE = 0.65                                 # regions are context-poor, so a species needs more
REGIONS = {(0, 0): "upper left", (1, 0): "upper right", (0, 1): "lower left", (1, 1): "lower right"}


def _scene_of(path: Path) -> tuple[str | None, str, float] | None:
    """(kind, readable label, probability) of the scene model's top guess for this picture or region,
    or None when the scene add-on is not installed."""
    try:
        import importlib.util

        from ..core.models import places_path
        if importlib.util.find_spec("ai_edge_litert") is None or not places_path().exists():
            return None
        from .scene_type import classify, kind_of, load_labels, readable
        p = classify(path, places_path())
        names, _ = load_labels()
        i = int(p.argmax())
        return kind_of(names[i]), readable(names[i]), float(p[i])
    except Exception:
        return None


def _is_built(scene, threshold: float) -> bool:
    return bool(scene) and scene[0] in BUILT and scene[2] >= threshold


def _crops(path: Path, workdir: Path) -> list[tuple[str, tuple[float, float, float, float], Path]]:
    """The 2x2 quarters and the centre of a picture, saved as files: a species is looked for in each
    region on its own, so a building in one part never hides a plant or animal in another."""
    out = []
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if min(W, H) < 160:
            return []
        boxes = [(f"the {REGIONS[(gx, gy)]} of the picture", (gx / 2, gy / 2, gx / 2 + 0.5, gy / 2 + 0.5))
                 for gy in (0, 1) for gx in (0, 1)]
        boxes.append(("the centre of the picture", (0.25, 0.25, 0.75, 0.75)))
        for k, (name, (x0, y0, x1, y1)) in enumerate(boxes):
            f = workdir / f"region{k}.png"
            im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))).save(f)
            out.append((name, (x0, y0, x1, y1), f))
    return out


class NatureIdProbe(Probe):
    name = "nature_id"
    group = "Living things"
    media = (MediaType.IMAGE,)
    frames_min_agree = 2     # video: the same species in two sampled frames, or it is only unconfirmed
    frames_ok = True         # video: the sampled frames are classified one by one (slow: a model load each)
    timeout = 300.0

    def run(self, path: Path) -> Finding:
        # A scan must never touch the network. These are read when huggingface_hub is first imported,
        # which files_present() below does, so they must be set before anything else runs.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not installed():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"Species identification is an optional add-on (about 1.7 GB): {INSTALL}")
        if not files_present():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="BioCLIP files not downloaded (scans never download): "
                                  "run: mitraav-setup --install nature")
        whole = None
        ctx = (self.context or {}).get("scene")
        if ctx and ctx.get("path") == str(path):         # the scene probe already looked at this picture
            whole = (ctx.get("kind"), ctx.get("label", ""), float(ctx.get("p", 0.0)))
        else:
            whole = _scene_of(path)
        whole_built = _is_built(whole, BUILT_P)
        from bioclip import Rank, TreeOfLifeClassifier
        clf = TreeOfLifeClassifier(model_str=MODEL_STR, device="cpu")
        found: list[dict] = []
        if not whole_built:
            found += self._look(clf, Rank, path, "the picture", None, MIN_SPECIES, allow_genus=True)
        workdir = Path(tempfile.mkdtemp(prefix="mitraav-nature-"))
        skipped = 0
        for name, box, f in _crops(path, workdir):
            sc = _scene_of(f)
            if _is_built(sc, TILE_BUILT_P):
                skipped += 1
                continue
            found += self._look(clf, Rank, f, name, box, MIN_TILE, allow_genus=False)
        if not found:
            if whole_built:
                return Finding(
                    probe=self.name, detected=False, info=True,
                    summary=f"The picture looks like **{whole[1]}** ({whole[2] * 100:.0f}%), a building or "
                            "street scene, and no region of it scored clearly as a plant, animal or fungus "
                            "(BioCLIP names a species for almost any picture, so only regions that are not "
                            "clearly buildings are trusted).")
            return Finding(probe=self.name, detected=False,
                           summary="Nothing scored clearly as a plant, animal or fungus (it may be small, far "
                                   "away, or not the main subject; people are not analysed).")
        best = max(found, key=lambda x: x["strength"] + (0.02 if x["box"] is None else 0.0))   # the whole picture wins ties
        where = "" if best["box"] is None else f" in {best['where']}"
        if whole_built and best["box"] is not None:
            where += (f" (the picture as a whole looks like {whole[1]}, so the rest was not trusted)")
        fin = best["finding"]
        fin.meta = {**fin.meta, "label": best["label"]}
        if where:
            key = " is consistent with" if " is consistent with" in fin.summary else " looks like"
            fin.summary = fin.summary.replace(key, where + key, 1)
        if best["box"] is not None:
            x0, y0, x1, y1 = best["box"]
            fin.boxes = [dict(label=best["label"], x0=x0, y0=y0, x1=x1, y1=y1)]
        return fin

    def _look(self, clf, Rank, path: Path, where: str, box, threshold: float, allow_genus: bool) -> list[dict]:
        """Species candidates for one picture or region, as dicts holding a finished Finding."""
        top = clf.predict(str(path), Rank.SPECIES, k=5)
        people = any(str(p.get("species", "")).lower() in SKIP or str(p.get("genus", "")).lower() in SKIP
                     for p in top[:1])
        cands = [p for p in top if str(p.get("species", "")).lower() not in SKIP
                 and str(p.get("genus", "")).lower() not in SKIP]
        if people or not cands:
            return []
        fin = self._finding(cands, threshold, allow_genus)
        if fin is None:
            return []
        label = (cands[0].get("species") or cands[0].get("genus") or "").strip()
        return [dict(finding=fin, strength=fin.strength, where=where, box=box, label=label)]

    def _finding(self, cands: list[dict], threshold: float, allow_genus: bool) -> Finding | None:
        best = cands[0]
        score = float(best.get("score", 0.0))
        if score < threshold:
            genus = best.get("genus")
            gscore = sum(float(p.get("score", 0)) for p in cands if p.get("genus") == genus)
            if allow_genus and genus and gscore >= MIN_GENUS:
                names = ", ".join(f"**{p.get('species')}** ({float(p.get('score', 0)) * 100:.0f}%)"
                                  for p in cands if p.get("genus") == genus)
                fu = [FollowUp(name=f"GBIF: {genus}", url=f"https://www.gbif.org/species/search?q={quote_plus(genus)}",
                               note="Occurrence map for the genus; compare the candidate species' ranges.")]
                return Finding(
                    probe=self.name, detected=True, strength=min(0.6, gscore), narrows=0.2,
                    constrains=["region", "season"], effort=2, follow_ups=fu,
                    summary=f"The {_kingdom_word(best.get('kingdom', ''))} looks like the genus **{genus}** "
                            f"(candidates: {names}); the species is uncertain. A genus gives only a "
                            "broad range. BioCLIP scores are leads, not identifications.")
            return None
        sci = best.get("species", "")
        common = best.get("common_name") or ""
        alts = ", ".join(f"{p.get('species')} ({float(p.get('score', 0)) * 100:.0f}%)" for p in cands[1:3]
                         if float(p.get("score", 0)) >= 0.1)
        q = quote_plus(sci)
        fu = [FollowUp(name=f"GBIF: {sci}", url=f"https://www.gbif.org/species/search?q={q}",
                       note="Occurrence map; the range says where it grows, not where the photo was taken."),
              FollowUp(name=f"iNaturalist: {sci}", url=f"https://www.inaturalist.org/taxa/search?q={q}",
                       note="Photos, range and seasonality to compare with your picture."),
              FollowUp(name="Plants of the World Online", url=f"https://powo.science.kew.org/results?q={q}",
                       note="For plants: native and introduced ranges.")]
        return Finding(
            probe=self.name, detected=True, strength=min(1.0, score), narrows=0.3,
            constrains=["region", "season"], effort=2, follow_ups=fu,
            summary=f"The {_kingdom_word(best.get('kingdom', ''))} is consistent with **{sci}**"
                    + (f" (**{common}**)" if common else "") + f" at {score * 100:.0f}%"
                    + (f"; other candidates: {alts}" if alts else "")
                    + ". Look-alike species are common and cultivated plants grow far outside their "
                      "native range, so use it as a lead to a range and season, not as a location.")
