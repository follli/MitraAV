"""Birdsong: BirdNET species detections, as a lead to species ranges and recordings.

Optional: needs `pip install "mitraav[birds]"` and the BirdNET model downloaded once (a scan never
downloads anything). A species list is a lead for region and season, not a location.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

from ..core.audio import AudioError, total_seconds
from ..core.birdworker import geo_model_path, human_segments, model_path, summarise
from ..core.finding import Finding, FollowUp, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

MARK = "@@JSON@@"
INSTALL = "install it from Add-ons in the app, or run: mitraav-setup --install birds"


class BirdProbe(Probe):
    name = "birdsong"
    group = "Sounds"
    clutter_sensitive = True
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 900.0

    def run(self, path: Path) -> Finding:
        if importlib.util.find_spec("birdnet") is None:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"Bird ID needs the optional BirdNET package: {INSTALL}")
        mp = model_path()
        if mp is None or not mp.exists():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"BirdNET model not downloaded (scans never download): {INSTALL}")
        try:
            total_seconds(path)
            r = subprocess.run([sys.executable, "-m", "mitraav.core.birdworker", str(path)],
                               capture_output=True, text=True, timeout=self.timeout - 30)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        except subprocess.TimeoutExpired:
            return Finding(probe=self.name, detected=False, status="timeout",
                           detail="BirdNET took too long on this file")
        if r.returncode != 0 or MARK not in r.stdout:
            return Finding(probe=self.name, detected=False, status="error",
                           detail=f"BirdNET failed: {r.stderr.strip()[-300:]}")
        payload = json.loads(r.stdout.split(MARK, 1)[1])
        sp = summarise(payload["dets"])
        humans = human_segments(payload["dets"])
        geo = payload.get("geo")
        if not sp:
            return Finding(probe=self.name, detected=False,
                           summary="No bird species scored credibly (>= 0.5 over two segments, or "
                                   ">= 0.9 once). Wind, insects and distance hide birds."
                                   + (f" BirdNET heard human voices in **{humans} segments**, "
                                      "which hides quiet bird calls; a recording with less speech "
                                      "is needed to tell." if humans >= 3 else ""))
        top = sp[:8]
        strength = min(1.0, 0.3 + 0.15 * min(len(top), 3) + 0.25 * top[0]["best"])
        names = [f"**{s['common']}** ({s['sci']}): {s['n']} segment{'s' if s['n'] != 1 else ''}, best "
                 f"{s['best']:.2f}, {s['t0']:.0f}-{s['t1']:.0f} s" for s in top]
        fu = follow_ups("audio", "birds")
        for s in top[:3]:
            q = quote_plus(s["sci"])
            fu.append(FollowUp(name=f"Xeno-canto: {s['common']}",
                               url=f"https://xeno-canto.org/explore?query={q}",
                               note="Compare recordings and see where the species is recorded."))
            fu.append(FollowUp(name=f"GBIF: {s['common']}",
                               url=f"https://www.gbif.org/species/search?q={q}",
                               note="Occurrence map; intersect the ranges of all detected species."))
        geo_note = ""
        if geo and geo.get("regions") and humans >= 10 and len(sp) < 3:
            geo_note = ("range scoring is not shown: with this much speech the few bird IDs are too "
                        "uncertain to say where they fit")
        elif geo and geo.get("regions"):
            rs = geo["regions"]
            best = rs[0]
            also = [r for r in rs[1:3] if r["fit"] >= 0.37]
            when = ""
            if best["seasons"] and "year-round" not in best["seasons"]:
                when = f", most clearly in {' or '.join(best['seasons'])}"
            geo_note = ("BirdNET's range model finds these species most consistent with **"
                        + best["region"] + "**" + when)
            if also:
                geo_note += ("; " + " and ".join(r["region"] for r in also)
                             + " also fit nearly as well (species with wide ranges fit several regions)")
            clear = not also and len(sp) >= 3
            geo_note += (". One region clearly fits best" if clear else
                         ". Treat this as a general region lead only")
        elif geo_model_path() is None or not geo_model_path().exists():
            geo_note = ("range scoring skipped because the BirdNET geo model is not downloaded "
                        "(download it once with: python -c \"import birdnet; birdnet.load('geo', '2.4', 'tf', "
                        "library='litert')\")")
        regions = [dict(label=f"{s['common']} {s['t0']:.0f}-{s['t1']:.0f} s",
                        t0=max(0.0, s["t0"] - 2), t1=s["t1"] + 2, f0=500.0, f1=10000.0)
                   for s in top]
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.35,
            constrains=["region", "season"], effort=2, regions=regions, follow_ups=fu,
            summary="BirdNET scores suggest: " + "; ".join(names)
            + (f". BirdNET also heard human voices in **{humans} segments**, which can hide quiet "
               "bird calls" if humans >= 3 else "")
            + (". " + geo_note if geo_note else "")
            + ". Species IDs from audio are leads, not proof (look-alike calls, and BirdNET "
              "covers 6500 species worldwide); intersect the ranges and the season, and "
              "confirm against recordings.")
