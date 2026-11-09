"""Offline speech language identification with Whisper (the `speech` add-on: faster-whisper, tiny model).

Only the language, how much speech there is, and non-sensitive clues in what was said (dates, domains,
currency and unit words, coordinates) are reported. The transcript itself is never stored or shown: it can
be a private conversation, and a lead about *where* only needs the language and the place clues.
"""
from __future__ import annotations

import importlib.util
import os

MODEL = "tiny"
REPO = "Systran/faster-whisper-tiny"
FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
LANGS = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
         "nl": "Dutch", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
         "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish", "da": "Danish",
         "no": "Norwegian", "fi": "Finnish", "el": "Greek", "he": "Hebrew", "th": "Thai", "vi": "Vietnamese",
         "id": "Indonesian", "cs": "Czech", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "sr": "Serbian",
         "hr": "Croatian", "ca": "Catalan", "fa": "Persian", "ur": "Urdu", "bn": "Bengali", "ta": "Tamil",
         "sw": "Swahili", "ms": "Malay", "tl": "Tagalog", "cy": "Welsh", "ga": "Irish", "mt": "Maltese"}
# narrowly located languages -> macro-regions (widely spoken ones such as English, Spanish, French, Arabic are not hinted)
LANG_MACROS = {"ja": {"asia"}, "ko": {"asia"}, "zh": {"asia"}, "th": {"asia"}, "vi": {"asia"}, "hi": {"asia"},
               "id": {"asia", "oceania"}, "ms": {"asia"}, "tl": {"asia", "oceania"}, "tr": {"europe", "asia"},
               "ru": {"europe", "asia"}, "uk": {"europe"}, "pl": {"europe"}, "de": {"europe"}, "it": {"europe"},
               "nl": {"europe"}, "sv": {"europe"}, "da": {"europe"}, "no": {"europe"}, "fi": {"europe"}, "el": {"europe"},
               "cs": {"europe"}, "hu": {"europe"}, "ro": {"europe"}, "bg": {"europe"}, "sr": {"europe"}, "hr": {"europe"},
               "mt": {"europe"}, "cy": {"europe"}, "ga": {"europe"}, "ca": {"europe"}, "he": {"asia"}, "fa": {"asia"}}


def installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def files_present() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(REPO, f), str) for f in FILES[:3])
    except Exception:
        return False


def download() -> None:
    from faster_whisper import WhisperModel
    WhisperModel(MODEL, device="cpu", compute_type="int8")


def go_offline() -> None:
    """A scan must never touch the network: these are read when huggingface_hub is first imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def offline_model():
    """The tiny model, loaded without any network access."""
    go_offline()
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL, device="cpu", compute_type="int8", local_files_only=True)
