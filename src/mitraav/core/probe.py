from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .finding import Finding, MediaType


class Probe(ABC):
    name: str = "unnamed"
    group: str = "Other"   # how the GUI nests probes
    quiet: bool = False    # True: reports only what a user could see (shown as context, not a lead)
    order: int = 50        # lower runs first (sound_events runs first so busy audio is known)
    clutter_sensitive: bool = False   # unreliable when speech/music fills the recording
    masked_by: tuple[str, ...] = ()   # sound-event categories that can imitate this probe's signal
    media: tuple[MediaType, ...] = (MediaType.ANY,)
    timeout: float = 10.0  # seconds; the runner enforces this
    frames_ok: bool = True  # image probes may be run on sampled video frames
    context: dict = {}      # what earlier probes found (set by the runner; see core/context.py)

    @abstractmethod
    def run(self, path: Path) -> Finding:
        """Cheap test only: is the signal present, and is it worth a closer look?"""
