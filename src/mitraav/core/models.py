"""Where MitraAV keeps downloaded models, and the one-time download (never done during a scan)."""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
PLACES_URL = "https://huggingface.co/litert-community/Places365-ResNet18-LiteRT/resolve/main/places_fp16.tflite"
PLACES_SHA256 = "8eb6c7ee5ec21be435d6d11f24bd68e13199bb457ff4c01fb07bde213f289d39"
SETUP_HINT = "install it from Add-ons in the app, or run: mitraav-setup --install sounds scene"


def model_dir() -> Path:
    env = os.environ.get("MITRAAV_MODELS")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "mitraav" / "models"


def yamnet_path() -> Path:
    return model_dir() / "yamnet.tflite"


def places_path() -> Path:
    return model_dir() / "places365_resnet18_fp16.tflite"


def _fetch(url: str, dest: Path, sha256: str, name: str, progress=print) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        progress(f"{name} already present: {dest}")
        return dest
    progress(f"Downloading {name} to {dest} ...")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name} download failed its checksum; not installed")
    tmp.replace(dest)
    return dest


def download_yamnet(progress=print) -> Path:
    """Fetch YAMNet (about 4 MB, Apache-2.0) and verify its checksum."""
    return _fetch(YAMNET_URL, yamnet_path(), YAMNET_SHA256, "YAMNet", progress)


def download_places(progress=print) -> Path:
    """Fetch the Places365 ResNet18 scene classifier (22.8 MB; MIT code, CC BY pretrained weights:
    credit the Places Project, places2.csail.mit.edu) and verify its checksum."""
    return _fetch(PLACES_URL, places_path(), PLACES_SHA256, "Places365 scene model", progress)
