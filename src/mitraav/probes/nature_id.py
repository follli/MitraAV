"""Living things in a picture: plant, animal and fungus species candidates (BioCLIP).

Optional and heavy (see core/nature.py). The whole picture is classified, so it works best on a
close, well-framed photo of one organism, not a street scene with a tree in it. A species is a
lead to its range (GBIF, iNaturalist), which says where it grows, not where the photo was taken.
People are never reported.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

from ..core.finding import Finding, FollowUp, MediaType
from ..core.nature import INSTALL, MODEL_STR, files_present, installed
from ..core.probe import Probe

MIN_SPECIES = 0.50
MIN_GENUS = 0.60
SKIP = ("homo sapiens", "homo")


def _kingdom_word(k: str) -> str:
    return {"Plantae": "plant", "Animalia": "animal", "Fungi": "fungus"}.get(k, "organism")


def _scene_gate(path: Path) -> str | None:
    """BioCLIP names a species for almost any picture. If the scene model says the picture is a
    building or street scene, no organism is the subject, so say that instead of guessing."""
    try:
        import importlib.util

        from ..core.models import places_path
        if importlib.util.find_spec("ai_edge_litert") is None or not places_path().exists():
            return None
        from .scene_type import classify, kind_of, load_labels, readable
        p = classify(path, places_path())
        names, _ = load_labels()
        i = int(p.argmax())
        if p[i] >= 0.30 and kind_of(names[i]) in ("street", "worship", "landmark"):
            return (f"The picture looks like **{readable(names[i])}** ({p[i] * 100:.0f}%), a building or "
                    "street scene rather than a close view of a plant or animal, so no species is "
                    "suggested (BioCLIP will name a species for almost any picture).")
    except Exception:
        return None
    return None


class NatureIdProbe(Probe):
    name = "nature_id"
    group = "Living things"
    media = (MediaType.IMAGE,)
    frames_ok = False        # loading the model per video frame would be wasteful
    timeout = 300.0

    def run(self, path: Path) -> Finding:
        if not installed():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"Species identification is an optional add-on (about 1.7 GB): {INSTALL}")
        if not files_present():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="BioCLIP files not downloaded (scans never download): "
                                  "run: mitraav-setup --install nature")
        gate = _scene_gate(path)
        if gate:
            return Finding(probe=self.name, detected=False, info=True, summary=gate)
        os.environ["HF_HUB_OFFLINE"] = "1"                  # a scan must never touch the network
        from bioclip import Rank, TreeOfLifeClassifier
        clf = TreeOfLifeClassifier(model_str=MODEL_STR, device="cpu")
        top = clf.predict(str(path), Rank.SPECIES, k=5)
        cands = [p for p in top if str(p.get("species", "")).lower() not in SKIP
                 and str(p.get("genus", "")).lower() not in SKIP]
        if not cands:
            return Finding(probe=self.name, detected=False,
                           summary="No plant, animal or fungus stood out (people are not analysed).")
        best = cands[0]
        score = float(best.get("score", 0.0))
        if score < MIN_SPECIES:
            # fall back to the genus when several top species share it
            genus = best.get("genus")
            gscore = sum(float(p.get("score", 0)) for p in cands if p.get("genus") == genus)
            if genus and gscore >= MIN_GENUS:
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
            return Finding(probe=self.name, detected=False,
                           summary="Nothing scored clearly as a species or genus (the organism may "
                                   "be small, far away, or not the main subject).")
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
