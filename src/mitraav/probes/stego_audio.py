"""Hidden data in sound: LSB replacement in lossless 16-bit audio, data in near-silence, and carriers
above the range of hearing.

* In **lossless** 16-bit audio (WAV, FLAC, PCM tracks) the chi-square attack looks for a message that
  fills the lowest bits from the start of the recording, and a check of passages that are nearly silent
  looks for lowest-bit noise where digital silence should be.
* In **any** audio, narrow steady tones above 15 kHz (pilot tones, advertising "beacons", data carriers
  or just equipment whine) are reported: a hearing-range recording rarely has them.

Lossy audio (MP3, AAC, Vorbis) rewrites the lowest bits, so only the spectral check applies there. Echo
hiding, phase coding and spread-spectrum methods are not tested.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.lines import find_lines
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.stego_stats import pcm_chi_square_prefix, prefix_verdict, silence_lsb_activity

LOSSLESS_CODECS = ("pcm_s16le", "pcm_s16be", "flac", "alac", "wavpack", "tta", "ape")
MAX_SECONDS = 120


def read_pcm16(path: Path) -> tuple[np.ndarray, int, str] | tuple[None, None, str]:
    """(first channel as int16, sample rate, '') for a lossless 16-bit track, else (None, None, reason)."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() == 2 and w.getcomptype() == "NONE":
                sr, ch = w.getframerate(), w.getnchannels()
                raw = w.readframes(min(w.getnframes(), sr * MAX_SECONDS))
                return np.frombuffer(raw, dtype="<i2")[::ch].astype(np.int16), sr, ""
            if w.getsampwidth() != 2:
                return None, None, f"the track is {w.getsampwidth() * 8}-bit (the LSB tests cover 16-bit audio)"
    except (wave.Error, EOFError, OSError):
        pass
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        return None, None, "ffmpeg is needed to read this container"
    try:
        r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-print_format", "json",
                            "-show_streams", str(path)], capture_output=True, text=True, timeout=60)
        s = (json.loads(r.stdout or "{}").get("streams") or [None])[0]
    except (subprocess.TimeoutExpired, ValueError):
        s = None
    if not s:
        return None, None, "no audio track"
    codec = s.get("codec_name", "")
    if codec not in LOSSLESS_CODECS:
        return None, None, f"the audio is {codec or 'lossy'}-compressed, which rewrites the lowest bits"
    bits = int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0)
    if bits != 16:
        return None, None, f"the track is {bits or 'unknown'}-bit (the LSB tests cover 16-bit audio)"
    sr, ch = int(s.get("sample_rate", 44100)), int(s.get("channels", 1))
    try:
        p = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", str(MAX_SECONDS),
                            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"], capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, None, "decoding took too long"
    x = np.frombuffer(p.stdout, dtype="<i2")
    return (x[::ch].astype(np.int16), sr, "") if len(x) else (None, None, "the audio could not be decoded")


def high_carriers(path: Path, min_hz: float = 15000.0) -> list[tuple[float, float, float]]:
    """(frequency, start, end) of steady narrow tones above min_hz, over the first 10 minutes."""
    out: list[tuple[float, float, float]] = []
    try:
        for i, (off, x, sr) in enumerate(iter_chunks(path)):
            if i >= 2 or sr < 32000 or len(x) / sr < 3:
                continue
            for ln in find_lines(x, sr, off, fmin=min_hz, fmax=0.49 * sr, min_dur=1.0, thr=18.0):
                if ln.prom >= 20.0 and not any(abs(ln.f - f) < 0.01 * f and abs(ln.t0 - t0) < 1.0 for f, t0, _ in out):
                    out.append((ln.f, ln.t0, ln.t1))
    except AudioError:
        return []
    return sorted(out)[:12]


class StegoAudioProbe(Probe):
    name = "hidden_data_audio"
    group = "Hidden data"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 400.0

    def run(self, path: Path) -> Finding:
        notes: list[str] = []
        strength = 0.0
        x, sr, why = read_pcm16(path)
        skipped = ""
        if x is None:
            skipped = why
        else:
            pre, ctl = pcm_chi_square_prefix(x, steps=20, with_control=True)
            verdict = prefix_verdict(pre, control=ctl or None)
            dur = len(x) / sr
            if verdict and verdict[0] == "partial":
                notes.append(f"the lowest bit of the samples looks like random message bits for the first "
                             f"**{verdict[1] * 100:.0f}%** of the recording (about {verdict[1] * dur:.0f} s) and natural "
                             "after that, as LSB replacement from the start of a file leaves")
                strength = 0.8
            near, share = silence_lsb_activity(x, sr)
            if near >= 0.25 and 0.05 <= share <= 0.95:
                notes.append(f"{near:.1f} s of near-silence carries a random mix of 0 and +-1 sample values "
                             f"({share * 100:.0f}% non-zero) instead of exact digital silence: dither, or data hidden "
                             "in the lowest bit of the quiet passages")
                strength = max(strength, 0.4)
        carriers = high_carriers(path)
        if carriers:
            desc = ", ".join(f"**{f / 1000:.1f} kHz** ({t1 - t0:.0f} s from {t0:.0f} s)" for f, t0, t1 in carriers[:4])
            notes.append(f"steady narrow tone(s) above 15 kHz: {desc}. Consistent with a pilot tone, an ultrasonic "
                         "beacon or data carrier, a sync signal, or just equipment whine")
            strength = max(strength, 0.45)
        if not notes:
            tail = f" (LSB tests skipped: {skipped})" if skipped else ""
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No lowest-bit pattern, odd near-silence or steady tone above 15 kHz found" + tail
                           + ". Echo hiding, phase coding, spread-spectrum and adaptive methods are not tested, "
                             "so this is not proof that nothing is hidden.")
        regions = [dict(label=f"tone near {f / 1000:.1f} kHz", t0=max(0.0, t0 - 1), t1=t1 + 1,
                        f0=max(0.0, f - 1500), f1=f + 1500) for f, t0, t1 in carriers[:4]]
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.4, effort=2,
            constrains=["hidden content"], regions=regions, follow_ups=follow_ups("stego", "audio"),
            summary=" • ".join(notes))
