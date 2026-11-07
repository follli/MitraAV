"""Optional BioCLIP support (plant, animal and fungus identification): where its files live.

Heavy: PyTorch (use the CPU build), open_clip, about 600 MB of weights and about 850 MB of species
embeddings, all installed on request by the Add-ons dialog or `mitraav-setup --install nature`,
never by default and never during a scan.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MODEL_STR = "hf-hub:imageomics/bioclip"             # BioCLIP v1 (MIT); v2 is 1.7 GB, left out
MODEL_REPO, MODEL_FILE = "imageomics/bioclip", "open_clip_pytorch_model.bin"
DATA_REPO = "imageomics/TreeOfLife-10M"            # species name embeddings for v1
DATA_FILES = ("embeddings/txt_emb_species.npy", "embeddings/txt_emb_species.json")
INSTALL = "install it from Add-ons in the app, or run: mitraav-setup --install nature"


def installed() -> bool:
    return importlib.util.find_spec("bioclip") is not None


def _cached(repo: str, filename: str, repo_type: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo_id=repo, filename=filename, repo_type=repo_type)
        return Path(p) if isinstance(p, (str, Path)) else None
    except Exception:
        return None


def files_present() -> bool:
    """True when the weights and embeddings are already in the Hugging Face cache."""
    return (_cached(MODEL_REPO, MODEL_FILE, "model") is not None
            and all(_cached(DATA_REPO, f, "dataset") for f in DATA_FILES))


def download(progress=print) -> None:
    """One-time download (about 1.5 GB). Only called from mitraav-setup --nature."""
    from bioclip import TreeOfLifeClassifier
    progress("Downloading BioCLIP weights and species embeddings (about 1.5 GB) ...")
    TreeOfLifeClassifier(model_str=MODEL_STR, device="cpu")
