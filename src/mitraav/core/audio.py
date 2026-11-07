from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)   # extra RIFF chunks are harmless

MAX_SECONDS = 300          # per analysis chunk
MAX_TOTAL = 3 * 3600       # longest stretch decoded/scanned
SILENCE = 1e-5             # peak below this counts as digital silence


class AudioError(RuntimeError):
    """Audio could not be loaded (missing ffmpeg, no audio stream, bad file)."""


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    else:
        x = data.astype(np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def _read_wav(path: Path, offset: float = 0.0, duration: float = MAX_SECONDS):
    sr, data = wavfile.read(path, mmap=True)
    a = int(max(offset, 0.0) * sr)
    return _to_float(np.asarray(data[a:a + int(duration * sr)])), int(sr)


def _wav_path(path: Path) -> Path:
    """A WAV scipy can mmap: the file itself, else an ffmpeg-decoded cached copy."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            wavfile.read(path, mmap=True)
            return path
        except Exception:
            pass  # unusual WAV encoding: fall through to ffmpeg
    return _ffmpeg_to_wav(path)


def total_seconds(path: Path) -> float:
    try:
        sr, data = wavfile.read(_wav_path(path), mmap=True)
        return len(data) / sr
    except AudioError:
        raise
    except Exception as e:
        raise AudioError(f"Could not read audio: {e}")


def sound_start(path: Path, block: float = 5.0) -> float:
    """Seconds of leading digital silence to skip (0 if the file starts with sound)."""
    w = _wav_path(path)
    sr, data = wavfile.read(w, mmap=True)
    n = int(block * sr)
    for a in range(0, min(len(data), int(MAX_TOTAL * sr)), n):
        if np.abs(_to_float(np.asarray(data[a:a + n]))).max(initial=0.0) > SILENCE:
            return max(0.0, a / sr - 1.0)
    return 0.0


def _ffmpeg_to_wav(path: Path) -> Path:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("This format needs ffmpeg: sudo apt install ffmpeg")
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "mitraav-cache"
    cache.mkdir(exist_ok=True)
    out = cache / f"{key}.wav"
    if not out.exists():
        tmp = cache / f"{key}.part.wav"
        cmd = [exe, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1",
               "-t", str(MAX_TOTAL), "-c:a", "pcm_s16le", str(tmp)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise AudioError("ffmpeg timed out decoding the file")
        if r.returncode != 0 or not tmp.exists():
            raise AudioError(f"ffmpeg could not decode audio: {r.stderr.strip()[:200]}")
        tmp.replace(out)
    return out


def load_audio(path: Path, offset: float | None = None,
               duration: float = MAX_SECONDS) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate: `duration` seconds from `offset`.
    With offset=None, leading digital silence is skipped (see sound_start)."""
    path = Path(path)
    try:
        if offset is None:
            offset = sound_start(path)
        return _read_wav(_wav_path(path), offset, duration)
    except AudioError:
        raise
    except Exception as e:
        raise AudioError(f"Could not read audio: {e}")


def iter_chunks(path: Path, chunk: float = MAX_SECONDS, overlap: float = 30.0):
    """Yield (offset_s, samples, sr) over the whole file after leading silence. Silent chunks are
    skipped. Chunks overlap so an event on a boundary is seen whole at least once."""
    path = Path(path)
    try:
        w = _wav_path(path)
        total = min(total_seconds(path), MAX_TOTAL)
        start = sound_start(path)
        step = max(chunk - overlap, 1.0)
        off = start
        first = True
        while first or off < total - 5.0:            # a short clip is one chunk, never none
            first = False
            x, sr = _read_wav(w, off, chunk)
            if len(x) and np.abs(x).max() > SILENCE:
                yield off, x, sr
            if off + chunk >= total:
                break
            off += step
    except AudioError:
        raise
    except Exception as e:
        raise AudioError(f"Could not read audio: {e}")
