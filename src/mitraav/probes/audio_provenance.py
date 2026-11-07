"""Audio file history: how the recording was probably made, processed or trimmed.

Measures bandwidth cliffs (lossy codecs, upsampling, telephone band), dual-mono, clipping, digital
silence and abrupt noise-floor steps. These say how the file got here, not where it was recorded;
they help judge whether the sound can be trusted and which tools to open next.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

from ..core.audio import MAX_SECONDS, AudioError, load_audio, sound_start, total_seconds
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

BAND = 250.0          # Hz per band when looking for a cliff
MIN_DROP_DB = 20.0


def bandwidth_cliff(x: np.ndarray, sr: int) -> float | None:
    """Frequency (Hz) above which the long-term spectrum falls off a cliff to a flat floor."""
    f, p = welch(x, fs=sr, nperseg=int(min(len(x), 8192)))
    db = 10 * np.log10(p + 1e-20)
    nb = int((sr / 2) // BAND)
    if nb < 12:
        return None
    bands = np.array([np.median(db[(f >= i * BAND) & (f < (i + 1) * BAND)]) for i in range(nb)])
    best, best_i = 0.0, None
    for i in range(3, nb - 5):
        before = float(np.mean(bands[i - 2:i + 1]))
        after = bands[i + 2:]
        drop = before - float(np.mean(after[:3]))
        flat = float(np.std(after)) <= 5.0 if len(after) >= 3 else False
        if drop > best and flat and (i + 1) * BAND < 0.96 * sr / 2:
            best, best_i = drop, i
    if best_i is None or best < MIN_DROP_DB:
        return None
    return float((best_i + 1) * BAND)


def describe_cliff(fc: float, sr: int) -> str:
    base = f"the spectrum drops off a cliff near **{fc / 1000:.1f} kHz**"
    if fc < 4000:
        why = "consistent with narrowband telephone audio (about 300 to 3400 Hz)"
    elif fc < 7600:
        why = "consistent with wideband telephone or voice-chat audio (about 50 to 7000 Hz)"
    elif fc < 9500:
        why = "consistent with audio originally sampled at 16 kHz and later resampled"
    elif fc < 13500:
        why = "consistent with a low-bitrate lossy codec (roughly 64-96 kbps MP3/AAC) or a 22 kHz source"
    elif fc < 17500:
        why = "consistent with a lossy codec such as MP3 around 128 kbps or similar AAC"
    else:
        why = "consistent with a higher-bitrate lossy codec (roughly 192-320 kbps)"
    return f"{base}, {why}" + (f", although the file is sampled at {sr} Hz" if sr >= 32000 and fc < 14000 else "")


def channel_info(path: Path) -> tuple[int, bool | None]:
    """(channels, dual_mono). dual_mono is None when it cannot be tested cheaply."""
    try:
        if path.suffix.lower() == ".wav":
            sr, d = wavfile.read(path, mmap=True)
            if d.ndim == 1:
                return 1, None
            n = min(len(d), sr * 60)
            a = np.asarray(d[len(d) // 2 - n // 2: len(d) // 2 + n // 2], dtype=np.float64)
            if not np.abs(a).max():
                return d.shape[1], None
            return d.shape[1], bool(np.array_equal(a[:, 0], a[:, 1]))
    except Exception:
        pass
    exe = shutil.which("ffprobe")
    if exe:
        try:
            r = subprocess.run([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=channels", "-of", "json", str(path)],
                               capture_output=True, text=True, timeout=20)
            return int(json.loads(r.stdout)["streams"][0]["channels"]), None
        except Exception:
            pass
    return 0, None


def internal_silence(x: np.ndarray, sr: int, min_s: float = 1.0) -> list[tuple[float, float]]:
    """Runs of exact digital silence inside the audio, as (start, end) seconds."""
    z = np.abs(x) < 1e-6
    out, start = [], None
    for i in np.flatnonzero(np.diff(np.r_[False, z, False].astype(np.int8))):
        if start is None:
            start = i
        else:
            if (i - start) / sr >= min_s:
                out.append((start / sr, i / sr))
            start = None
    return out


def noise_floor_steps(x: np.ndarray, sr: int, jump_db: float = 12.0) -> list[float]:
    """Times where the background level (10th percentile of 50 ms frames) steps abruptly."""
    n = int(sr * 0.05)
    m = len(x) // n
    if m < 160:
        return []
    rms = np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    w = 40                                      # 2 s each side
    steps = []
    for i in range(w, m - w, 10):
        a = np.percentile(db[i - w:i], 10)
        b = np.percentile(db[i:i + w], 10)
        if abs(b - a) >= jump_db and not (steps and i * 0.05 - steps[-1] < 4.0):
            steps.append(i * 0.05)
    return steps[:4]


class AudioProvenanceProbe(Probe):
    name = "audio_provenance"
    group = "File and metadata"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 60.0

    def run(self, path: Path) -> Finding:
        try:
            lead = sound_start(path)
            x, sr = load_audio(path, offset=lead, duration=MAX_SECONDS)
            total = total_seconds(path)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if len(x) / sr < 3 or sr < 4000:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Need at least 3 s of audio sampled at 4 kHz or more")
        notes, cons = [], ["source type"]
        if lead >= 2.0:
            notes.append(f"the first **{lead:.0f} s** of the file are **digital silence** (the sound starts "
                         "late: trimmed, muted or the recording started late)")
        fc = bandwidth_cliff(x, sr)
        if fc:
            notes.append(describe_cliff(fc, sr))
        ch, dual = channel_info(Path(path))
        if dual:
            notes.append(f"both of the {ch} channels are identical (**dual mono**), so there is no "
                         "stereo information; usually a mono source converted to stereo")
        peak_frac = float((np.abs(x) >= 0.999).mean())
        if peak_frac > 1e-4:
            notes.append(f"{peak_frac * 100:.2f}% of samples sit at full scale (clipping or "
                         "heavy limiting)")
        gaps = [g for g in internal_silence(x, sr) if g[0] > 3.0]   # skip the start-of-sound edge
        if gaps:
            notes.append("**digital silence** inside the audio near "
                         + ", ".join(f"{lead + a:.0f}-{lead + b:.0f} s" for a, b in gaps[:4])
                         + " (a gate, mute or edit)")
        steps = [t for t in noise_floor_steps(x, sr) if t > 6.0]
        if steps:
            notes.append("the background level steps abruptly near "
                         + ", ".join(f"{lead + t:.0f} s" for t in steps)
                         + " (a cut, a splice, or a gate; natural events can do this too)")
        if total > MAX_SECONDS + lead:
            notes.append(f"only the first {MAX_SECONDS} s of sound were analysed")
        if not (fc or dual or gaps or steps or peak_frac > 1e-4 or lead >= 2.0):
            return Finding(probe=self.name, detected=False, info=True,
                           summary="Nothing unusual about the file's history: **full bandwidth**, no "
                                   "silent gaps, no abrupt noise-floor steps, no clipping. "
                                   f"Sampled at {sr} Hz, {ch or '?'} channel{'s' if ch != 1 else ''}.")
        strength = min(1.0, 0.25 + 0.15 * len(notes))
        if gaps or steps:
            cons.append("edit history")
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.15,
                       constrains=cons, effort=1, follow_ups=follow_ups("audio", "provenance"),
                       summary="; ".join(notes) + ". These describe how the file was made or "
                       "processed, not where it was recorded; check the original source if you can.")
