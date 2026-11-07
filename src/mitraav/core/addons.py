"""Optional add-ons: extra probes that need a package, a model download, or both.

One registry for the app (Add-ons dialog) and the command line (mitraav-setup). Each add-on knows
whether it is installed and how to install itself. Nothing here runs during a scan, and nothing is
installed unless the user asks (a scan never touches the network). BioCLIP is the heavy one and is
never installed by default.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from . import models, nature

Progress = Callable[[str], None]


@dataclass
class Step:
    label: str
    done: Callable[[], bool]
    command: list[str] | Callable[[], list[str]]   # run with the current Python


@dataclass
class Addon:
    key: str
    name: str
    adds: str                          # what it enables, in plain words
    size: str                          # what the user will download
    steps: list[Step] = field(default_factory=list)
    heavy: bool = False

    def status(self) -> str:
        """'installed', 'not installed' or 'partly installed' (some steps done)."""
        flags = [s.done() for s in self.steps]
        if all(flags):
            return "installed"
        return "not installed" if not any(flags) else "partly installed"

    def pending(self) -> list[Step]:
        return [s for s in self.steps if not s.done()]


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pip(*args: str) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _torch_ok() -> bool:
    """torch and torchvision installed as a matching CPU pair (a PyPI torchvision next to a CPU
    torch is a version mismatch that breaks `import torchvision`)."""
    from importlib import metadata
    try:
        tv, tt = metadata.version("torchvision"), metadata.version("torch")
    except metadata.PackageNotFoundError:
        return False
    return sys.platform != "linux" or (tv.endswith("+cpu") and tt.endswith("+cpu"))


def _torch_command() -> list[str]:
    if _has("torch") or _has("torchvision"):          # repair: reinstall the pair, leave the rest
        return pip("--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", CPU_INDEX)
    return pip("torch", "torchvision", "--index-url", CPU_INDEX)


def _birds_files() -> bool:
    try:
        from .birdworker import geo_model_path, model_path
        a, g = model_path(), geo_model_path()
        return bool(a and g and a.exists() and g.exists())
    except Exception:
        return False


def registry() -> list[Addon]:
    litert = Step("ai-edge-litert (runtime, about 20 MB)", lambda: _has("ai_edge_litert"),
                  pip("ai-edge-litert"))
    return [
        Addon("sounds", "Sound classifier",
              "Names sounds: trains, horns, bells, sirens, aircraft, birds, animals, weather.",
              "about 25 MB",
              [litert, Step("YAMNet model (4 MB)", lambda: models.yamnet_path().exists(),
                            py("from mitraav.core import models; models.download_yamnet()"))]),
        Addon("scene", "Scene recognition",
              "Names the kind of place in a picture (street, church, desert...), as quiet context.",
              "about 45 MB",
              [litert, Step("Places365 model (23 MB)", lambda: models.places_path().exists(),
                            py("from mitraav.core import models; models.download_places()"))]),
        Addon("birds", "Birdsong and range",
              "Identifies birds by sound and suggests a general region and season (BirdNET). The BirdNET models are published for non-commercial use.",
              "about 150 MB",
              [Step("BirdNET package", lambda: _has("birdnet"), pip("birdnet>=1.4")),
               Step("BirdNET models (about 120 MB)", _birds_files,
                    py("import birdnet\nbirdnet.load('acoustic','2.4','tf',library='litert')\n"
                       "birdnet.load('geo','2.4','tf',library='litert')"))]),
        Addon("nature", "Plants and animals (BioCLIP)",
              "Suggests species for a photo of a plant, animal or fungus, with range links. Heavy: CPU PyTorch, BioCLIP weights and species embeddings.",
              "about 1.7 GB",
              [Step("PyTorch and torchvision, matching CPU builds (about 250 MB)", _torch_ok,
                    _torch_command),
               Step("pybioclip package", nature.installed, pip("pybioclip>=2")),
               Step("BioCLIP weights and species embeddings (about 1.5 GB)", nature.files_present,
                    py("from mitraav.core import nature; nature.download()"))], heavy=True),
    ]


def by_key(key: str) -> Addon | None:
    return next((a for a in registry() if a.key == key), None)


def system_tools() -> list[tuple[str, bool, str]]:
    """(tool, found, how to get it): these need the operating system's package manager."""
    return [
        ("ffmpeg", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
         "sudo apt install ffmpeg   (video, non-WAV audio, video details)"),
        ("exiftool", bool(shutil.which("exiftool")),
         "sudo apt install libimage-exiftool-perl   (edit traces in images)"),
        ("tesseract", bool(shutil.which("tesseract")),
         "sudo apt install tesseract-ocr   (reading text in images; add tesseract-ocr-<lang> packs)"),
    ]


class Cancelled(RuntimeError):
    pass


def run_steps(addon: Addon, progress: Progress, should_stop: Callable[[], bool] = lambda: False,
              set_proc: Callable[[subprocess.Popen | None], None] = lambda p: None) -> None:
    """Run the missing steps in order. Output lines go to `progress`. Raises on failure."""
    for step in addon.pending():
        progress(f"== {step.label}")
        cmd = step.command() if callable(step.command) else step.command
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        set_proc(proc)
        buf = b""
        while True:
            chunk = proc.stdout.read1(512)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
            for ln in lines:
                text = ln.decode(errors="replace").strip()
                if text:
                    progress(text[:200])
            if should_stop():
                proc.terminate()
        proc.wait()
        set_proc(None)
        if should_stop():
            raise Cancelled("cancelled")
        if proc.returncode != 0:
            raise RuntimeError(f"step failed: {step.label} (exit code {proc.returncode})")
    importlib.invalidate_caches()
    progress(f"Done: {addon.name} is installed.")
