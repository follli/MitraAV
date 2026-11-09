"""Provenance and AI-generation clues in metadata: C2PA Content Credentials, AI tool names, generator
parameters and prompts stored in PNG text chunks, XMP source-type declarations and editor history.

Everything here is declared by the file, not measured from it, and is easy to strip or forge: a hit is a
strong clue, a miss means nothing. C2PA manifests are detected but not verified.
"""
from __future__ import annotations

from pathlib import Path

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.provenance import analyse
from ..core.references import follow_ups


class ProvenanceAiProbe(Probe):
    name = "provenance_ai"
    group = "File and metadata"
    media = (MediaType.ANY,)
    timeout = 60.0

    def run(self, path: Path) -> Finding:
        try:
            r = analyse(path)
        except OSError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        notes, strength = [], 0.0
        ups = []
        if r.c2pa:
            gen = f" (generator mentions: {', '.join(r.generators[:3])})" if r.generators else ""
            notes.append(f"a **C2PA Content Credentials** manifest is embedded in the {r.c2pa_where}{gen}; it is "
                         "detected here, not verified")
            strength = max(strength, 0.5)
            ups += follow_ups("provenance", "c2pa")
        if r.declares_ai:
            tools = r.ai_tools[:3] or ([r.gen_source] if r.gen_source else [])
            kinds = [s for s in r.source_types if "lgorithmic" in s or "AI" in s]
            notes.append("the metadata declares **AI-generated or AI-assisted** content"
                         + (f" (tool names: {', '.join(tools)})" if tools else "")
                         + (f" (source type: {', '.join(kinds[:2])})" if kinds else ""))
            strength = max(strength, 0.9)
            ups += follow_ups("provenance", "ai")
        if r.gen_params:
            ps = ", ".join(f"{k} {v}" for k, v in list(r.gen_params.items())[:5])
            notes.append(f"generator settings are stored in the file ({ps})")
        if r.prompt:
            notes.append(f"a text prompt is stored in the file: \"{r.prompt[:100]}{'...' if len(r.prompt) > 100 else ''}\"")
        notes += r.notes
        if r.editors:
            notes.append(f"the metadata names editing software: {', '.join(r.editors[:4])} (the picture or video was "
                         "processed after capture)")
            strength = max(strength, 0.3)
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No C2PA manifest, AI-generator tag, stored prompt or editing-software name in the "
                                   "metadata. Metadata is easy to strip, so this says little.")
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.2, effort=1,
                       constrains=["origin"], follow_ups=ups,
                       summary=" • ".join(notes) + " • Declared by the file itself: easy to remove or forge.")
