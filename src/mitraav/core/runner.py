from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import pkgutil
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .finding import Finding, MediaType
from .frames import extract_frames, ffprobe
from .probe import Probe

BUSY_THRESHOLD = 0.35   # share of the recording that is speech or music

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".bmp", ".gif"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aif", ".aiff",
             ".amr", ".mka", ".ac3", ".wv"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ogv", ".wmv", ".flv", ".3gp",
             ".mpg", ".mpeg", ".ts", ".mts", ".m2ts"}
SNIFF_EXT = {".ogg", ".ogx"}        # Ogg can hold audio or video; the name does not say which
IMAGE_CODECS = {"mjpeg", "png", "webp", "bmp", "gif", "tiff", "jpeg2000", "heic", "hevc_image"}


@lru_cache(maxsize=64)
def _sniff(path: str, mtime: float) -> MediaType | None:
    """Classify a file by what ffprobe finds inside it (used for Ogg and unknown extensions)."""
    info = ffprobe(Path(path))
    streams = info.get("streams", [])
    if not streams:
        return None
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if video and not audio and video[0].get("codec_name") in IMAGE_CODECS:
        return MediaType.IMAGE
    if video:
        return MediaType.VIDEO
    return MediaType.AUDIO if audio else None


def media_of(path: Path) -> MediaType | None:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return MediaType.IMAGE
    if ext in AUDIO_EXT:
        return MediaType.AUDIO
    if ext in VIDEO_EXT:
        return MediaType.VIDEO
    if ext in SNIFF_EXT or ext not in IMAGE_EXT | AUDIO_EXT | VIDEO_EXT:
        try:
            return _sniff(str(path), path.stat().st_mtime)
        except OSError:
            return None
    return None


def discover_probes() -> list[type[Probe]]:
    import mitraav.probes as pkg

    found = []
    for m in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
        for obj in vars(mod).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Probe)
                and obj.__module__ == mod.__name__
                and not inspect.isabstract(obj)
            ):
                found.append(obj)
    return found


def _worker(conn, probe_cls, path):
    try:
        conn.send(probe_cls().run(path))
    except Exception as e:  # report, never crash the scan
        conn.send(Finding(probe=probe_cls.name, detected=False, status="error",
                          detail=f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_probe(probe_cls: type[Probe], path: Path) -> Finding:
    """Run one probe in its own process so a hang can be killed."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, probe_cls, path), daemon=True)
    proc.start()
    child.close()
    try:
        if parent.poll(probe_cls.timeout):
            result = parent.recv()
        else:
            result = Finding(probe=probe_cls.name, detected=False, status="timeout",
                             detail=f"No result within {probe_cls.timeout:g}s")
    except EOFError:
        result = Finding(probe=probe_cls.name, detected=False, status="error",
                         detail="Probe process died")
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(1)
        parent.close()
    return replace(result, group=probe_cls.group, quiet=probe_cls.quiet)

def run_on_frames(cls: type[Probe], path: Path) -> Finding:
    """Run an image probe on sampled video frames; report the strongest result."""
    frames = extract_frames(path)
    if not frames:
        return Finding(probe=cls.name, group=cls.group, quiet=cls.quiet, detected=False, status="not_run",
                       detail="Could not extract video frames (is ffmpeg installed?)")
    results = [(t, run_probe(cls, fp)) for t, fp in frames]
    ok = [r for r in results if r[1].status == "ok"]
    t, best = max(ok or results, key=lambda r: r[1].score)
    if best.detected:
        hits = sum(1 for _, f in ok if f.detected)
        best = replace(best, summary=f"[frame at {t:.0f}s; {hits}/{len(ok)} sampled frames "
                                     f"flagged] {best.summary}")
    return best


MASK_THRESHOLD = 0.30   # share of the recording a masking sound class must fill
CATEGORY_NAMES = {"sirens": "a siren or alarm", "music": "music", "voices": "speech", "bells": "bells"}


def masking_caveat(cls: type[Probe], shares: dict[str, float]) -> str | None:
    """A probe is inconclusive when the sound classifier hears something that imitates its signal:
    a siren's sweeping pitch and swelling volume look like glides, pass-bys and tones."""
    for cat in cls.masked_by:
        if shares.get(cat, 0.0) >= MASK_THRESHOLD:
            return (f"The sound classifier hears {CATEGORY_NAMES.get(cat, cat)} for about "
                    f"{shares[cat] * 100:.0f}% of this recording, which can imitate what this probe "
                    "looks for (for a siren: sweeping pitch and swelling volume look like glides, "
                    "pass-bys and tones), so the result is inconclusive.")
    return None


def planned(path: Path) -> list[tuple[type[Probe], str]]:
    """The probes a scan of this file will run, in order, as (class, 'direct' | 'frames')."""
    media = media_of(path)
    out = []
    for cls in sorted(discover_probes(), key=lambda c: c.order):
        if MediaType.ANY in cls.media or media in cls.media:
            out.append((cls, "direct"))
        elif media == MediaType.VIDEO and MediaType.IMAGE in cls.media and cls.frames_ok:
            out.append((cls, "frames"))
    return out


def scan(path: Path, on_result: Callable[[Finding], None] | None = None,
         on_start: Callable[[str], None] | None = None, only: set[str] | None = None,
         context: dict | None = None) -> list[Finding]:
    """Run the planned probes, or just those named in `only`. `context` carries what an earlier
    sound overview found ({"busy": share, "shares": {...}}) so a partial run can still judge masking;
    when a partial run needs it and has none, the sound overview is run first."""
    findings = []
    busy: float | None = (context or {}).get("busy")
    shares: dict[str, float] = dict((context or {}).get("shares") or {})
    todo = planned(path)
    if only is not None:
        chosen = [(c, m) for c, m in todo if c.name in only]
        needs = any(c.clutter_sensitive or c.masked_by for c, _ in chosen)
        if needs and busy is None and not any(c.name == "sound_events" for c, _ in chosen):
            chosen = [(c, m) for c, m in todo if c.name == "sound_events"] + chosen
        todo = chosen
    for cls, mode in todo:
        if on_start:
            on_start(cls.name)
        f = run_probe(cls, path) if mode == "direct" else run_on_frames(cls, path)
        if f.meta.get("busy") is not None:
            busy = float(f.meta["busy"])
            shares = dict(f.meta.get("shares") or {})
        if not f.caveat and f.status == "ok":
            note = masking_caveat(cls, shares)
            if note:
                f = replace(f, caveat=note)
        if busy is not None and busy >= BUSY_THRESHOLD and cls.clutter_sensitive and f.status == "ok":
            if f.detected:       # a lead found despite speech is kept (faint signals matter), with a note
                f = replace(f, summary=f.summary + f" (Speech or music fills about {busy * 100:.0f}% of "
                                                   "this recording, so confirm it on a cleaner stretch.)")
            else:
                f = replace(f, caveat=f"Speech or music fills about {busy * 100:.0f}% of this recording, "
                                      "which masks faint sounds and can imitate tones, so this result is "
                                      "inconclusive: absence proves nothing and a lead needs a clean "
                                      "recording to confirm.")
        findings.append(f)
        if on_result:
            on_result(f)
    findings.sort(key=lambda f: (-f.score, f.status != "ok", f.probe))
    return findings
