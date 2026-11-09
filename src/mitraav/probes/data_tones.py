"""Tone-coded signals inside audio: DTMF key presses, Morse code, SSTV picture headers and 1200-baud
AFSK data bursts (packet radio, APRS).

These are the "overlooked" signals in recordings of phone lines, radio and some hidden-message clips:
a person hears beeps, a tool can read them. DTMF digits are never printed (they may be a phone number
or a PIN): only that key presses occurred and when. Morse text is decoded when it is clean, and is
worded as a decode, not a transcript.
"""
from __future__ import annotations

from pathlib import Path

from ..core.audio import AudioError, iter_chunks
from ..core.datatones import afsk1200, downsample, dtmf, morse, sstv
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

MAX_CHUNKS = 3


class DataTonesProbe(Probe):
    name = "data_tones"
    group = "Hidden data"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 400.0

    def run(self, path: Path) -> Finding:
        notes: list[str] = []
        regions: list[dict] = []
        strength = 0.0
        seen = set()
        try:
            for i, (off, x, sr) in enumerate(iter_chunks(path)):
                if i >= MAX_CHUNKS:
                    break
                if sr < 8000 or len(x) / sr < 1.5:
                    continue
                x, sr = downsample(x, sr, 16000)
                keys = dtmf(x, sr)
                if keys and "dtmf" not in seen:
                    seen.add("dtmf")
                    t0, t1 = off + keys[0]["t"], off + keys[-1]["t"] + keys[-1]["dur"]
                    notes.append(f"**{len(keys)} DTMF key presses** (touch-tone dialling) between {t0:.0f} and {t1:.0f} s; "
                                 "the digits are not shown (they may be a phone number or PIN)")
                    regions.append(dict(label=f"DTMF keys near {t0:.0f} s", t0=max(0.0, t0 - 1), t1=t1 + 1, f0=600.0, f1=1700.0))
                    strength = max(strength, 0.7)
                m = morse(x, sr)
                if m and "morse" not in seen:
                    seen.add("morse")
                    shown = m["text"][:60]
                    notes.append(f"keyed-tone **Morse code** at about {m['tone']:.0f} Hz, roughly {m['wpm']:.0f} words per minute, "
                                 f"decodes to \"{shown}\" ({m['valid'] * 100:.0f}% of the letters valid; a decode, not a transcript)")
                    regions.append(dict(label=f"Morse near {off + m['t0']:.0f} s", t0=max(0.0, off + m["t0"] - 1),
                                        t1=off + m["t1"] + 1, f0=max(0.0, m["tone"] - 300), f1=m["tone"] + 300))
                    strength = max(strength, 0.8)
                for h in sstv(x, sr):
                    if "sstv" in seen:
                        break
                    seen.add("sstv")
                    notes.append(f"an **SSTV (slow-scan television)** calibration header near {off + h['t']:.0f} s"
                                 + (f", mode **{h['mode']}**" if h["mode"] else "")
                                 + ": a picture is likely encoded in the sound that follows")
                    regions.append(dict(label=f"SSTV header near {off + h['t']:.0f} s", t0=max(0.0, off + h["t"] - 1),
                                        t1=off + h["t"] + 120, f0=1000.0, f1=2500.0))
                    strength = max(strength, 0.85)
                bursts = afsk1200(x, sr)
                if bursts and "afsk" not in seen:
                    seen.add("afsk")
                    notes.append(f"**{len(bursts)} burst(s) of 1200-baud AFSK** (1200/2200 Hz tone pairs, packet radio or APRS), "
                                 f"first near {off + bursts[0]['t0']:.0f} s")
                    regions.append(dict(label=f"AFSK near {off + bursts[0]['t0']:.0f} s", t0=max(0.0, off + bursts[0]["t0"] - 1),
                                        t1=off + bursts[0]["t1"] + 1, f0=1000.0, f1=2500.0))
                    strength = max(strength, 0.7)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not notes:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No DTMF dialling, Morse code, SSTV header or 1200-baud AFSK found in the first minutes "
                                   "of sound. Other modes (RTTY, POCSAG, fax, spread-spectrum) are not tested.")
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.4, effort=2,
                       constrains=["hidden content"], regions=regions, follow_ups=follow_ups("stego", "datatones"),
                       summary=" • ".join(notes) + " • Confirm by ear and with a decoder: speech, music and "
                       "whistles can imitate short tone patterns.")
