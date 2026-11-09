"""Loudest moment: a minimal example plugin probe (copy it, rename it, make it yours).

It finds the loudest one-second stretch of an audio or video soundtrack and says when it is. It shows
every piece a plugin needs: a Probe subclass, the media it accepts, a Finding with a summary,
a time region (which gives the row a Spectro button and play-from-here links) and follow-up links.
The full guide is docs/PLUGINS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mitraav.plugin import AudioError, Finding, FollowUp, MediaType, Probe, iter_chunks

MITRAAV_API = 1                      # the plugin API version this file was written for


class LoudestMomentProbe(Probe):
    name = "loudest_moment"          # unique, lower case; shown in the table as "loudest moment"
    group = "Sounds"                 # which group the row is listed under
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 120.0                  # seconds: the runner stops the probe if it takes longer

    def run(self, path: Path) -> Finding:
        best_level, best_time = -1e9, 0.0
        try:
            for offset, x, sr in iter_chunks(path):          # long files arrive in 5-minute chunks
                win = int(sr)                                # one second
                if len(x) < win:
                    continue
                n = len(x) // win
                rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
                i = int(np.argmax(rms))
                level = 20 * np.log10(rms[i] + 1e-12)
                if level > best_level:
                    best_level, best_time = float(level), offset + i
        except AudioError as e:                              # no usable audio is "not run", not a crash
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if best_level < -80:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="The recording is silent or nearly so.")
        return Finding(
            probe=self.name, detected=True, strength=0.3, narrows=0.05,
            summary=f"The loudest second is near **{int(best_time // 60)}:{int(best_time % 60):02d}** "
                    f"(about {best_level:.0f} dBFS RMS).",
            regions=[dict(label=f"loudest second near {best_time:.0f} s", t0=max(0.0, best_time - 2),
                          t1=best_time + 3, f0=0.0, f1=8000.0, t_play=max(0.0, best_time - 2))],
            follow_ups=[FollowUp("Audacity", "https://www.audacityteam.org",
                                 "Open the file to look at that moment in detail.")])
