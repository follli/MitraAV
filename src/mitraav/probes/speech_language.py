"""Which language is spoken, and how much speech is there? (Whisper tiny, the `speech` add-on.)

Language is one of the strongest cheap location clues in a recording. This probe runs Whisper's language
identification on the speech found in the first few minutes, reports the language with its probability,
how much speech there is, and any place-like clues in what was said (dates, web domains, currency and
unit words, coordinates). It does not output the transcript. Whisper tiny is small and fast: it mistakes
related languages and can invent a language for music or noise, so a low probability is called out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ..core import speech
from ..core.audio import AudioError, load_audio
from ..core.finding import Finding, MediaType
from ..core.geo import hint
from ..core.probe import Probe
from .image_text import clues

SECONDS = 240


class SpeechLanguageProbe(Probe):
    name = "speech_language"
    group = "Sounds"
    clutter_sensitive = False
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 900.0

    def run(self, path: Path) -> Finding:
        speech.go_offline()                                # before anything imports huggingface_hub
        if not speech.installed():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Speech language identification is an add-on: install it from Add-ons in the app, or run: "
                                  "mitraav-setup --install speech")
        if not speech.files_present():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="The Whisper model is not downloaded (scans never download): run: mitraav-setup --install speech")
        try:
            x, sr = load_audio(path, duration=SECONDS)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if len(x) / sr < 3:
            return Finding(probe=self.name, detected=False, status="not_run", detail="Need at least 3 s of audio")
        if sr != 16000:
            g = np.gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g)
        model = speech.offline_model()
        segments, info = model.transcribe(x.astype(np.float32), vad_filter=True, beam_size=1,
                                          condition_on_previous_text=False)
        words, secs, n = [], 0.0, 0
        for seg in segments:
            n += 1
            secs += max(0.0, seg.end - seg.start)
            words += seg.text.split()
        if secs < 4 or len(words) < 3:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="Little or no speech found in the first minutes (under about 4 s of speech), so no "
                                   "language can be judged.")
        lang = speech.LANGS.get(info.language, info.language)
        p = float(info.language_probability)
        c = clues([(w, 100) for w in words])
        notes = [f"the speech (about {secs:.0f} s in {n} segments, roughly {len(words)} words) is most likely **{lang}** "
                 f"({p * 100:.0f}% confidence)"
                 + ("; this small model is unsure, so treat it as a guess" if p < 0.6 else "")]
        for stamp in c["timestamps"][:2]:
            notes.append(f"the speech mentions the date/time **{stamp}**")
        regs = sorted({w for key in ("domain", "currency", "units") for _, w in c[key]})
        if regs:
            notes.append("place clues in what was said: " + ", ".join(regs[:3]))
        macros = speech.LANG_MACROS.get(info.language, set()) if p >= 0.6 else set()
        return Finding(probe=self.name, detected=True, strength=min(0.9, 0.2 + 0.7 * p), narrows=0.3, effort=2,
                       constrains=["language", "region"],
                       meta={"geo": hint(f"spoken language: {lang}", macros, 0.15)} if macros else {},
                       summary=" • ".join(notes) + " • The transcript is not stored or shown. A language marks a "
                       "community, not a place: speakers travel and recordings are shared.")
