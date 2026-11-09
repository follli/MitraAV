"""The public API for plugin probes. Import from here, not from mitraav.core, so a plugin keeps
working as the inside of MitraAV changes. See docs/PLUGINS.md."""
from __future__ import annotations

from .core.audio import AudioError, iter_chunks, load_audio, total_seconds
from .core.finding import Finding, FollowUp, MediaType
from .core.geo import hint as geo_hint
from .core.plugins import API_VERSION
from .core.probe import Probe

__all__ = ["API_VERSION", "AudioError", "Finding", "FollowUp", "MediaType", "Probe", "geo_hint",
           "iter_chunks", "load_audio", "total_seconds"]
