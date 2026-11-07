from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class MediaType(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    ANY = "any"


def plain(text: str) -> str:
    """Summaries mark key words with **double asterisks** (the GUI shows them bold)."""
    return text.replace("**", "")


@dataclass
class FollowUp:
    name: str
    url: str = ""
    note: str = ""


@dataclass
class Finding:
    probe: str
    detected: bool
    strength: float = 0.0          # 0..1: how clear the signal is
    narrows: float = 0.0           # 0..1: how much it narrows the search
    constrains: list[str] = field(default_factory=list)  # e.g. ["region", "date"]
    summary: str = ""              # calibrated wording: "consistent with...", not "is"
    follow_ups: list[FollowUp] = field(default_factory=list)
    effort: int = 2                # follow-up effort: 1 cheap .. 3 heavy
    status: str = "ok"             # ok | not_run | error | timeout
    detail: str = ""
    # where the signal is, for viewers: [{"label", "t0", "t1" (file seconds), "f0", "f1" (Hz)}]
    regions: list[dict] = field(default_factory=list)
    group: str = ""                # set by the runner from the probe class
    quiet: bool = False            # context only: things a user could see for themselves
    info: bool = False             # ran fine, nothing wrong or unusual: shown as "info", not "nothing"
    pill: str = ""                 # optional custom result label for the table (e.g. "hash created")
    caveat: str = ""               # set by the runner: why this result is unreliable (busy audio)
    meta: dict = field(default_factory=dict)   # structured extras (e.g. {"busy": 0.6})

    @property
    def score(self) -> float:
        if not self.detected or self.status != "ok":
            return 0.0
        return self.strength * self.narrows / (1 + 0.25 * (self.effort - 1))

    @property
    def confidence_label(self) -> str:
        """How clearly the signal is present (not how useful it is for locating)."""
        if not self.detected or self.status != "ok":
            return "-"
        return "clear" if self.strength >= 0.7 else "likely" if self.strength >= 0.4 else "faint"

    @property
    def inconclusive(self) -> bool:
        return bool(self.caveat) and self.status == "ok"

    @property
    def very_low(self) -> bool:
        """A faint lead worth a flag. File-history probes describe the file, not a location lead."""
        return (self.detected and self.status == "ok" and self.group != "File and metadata"
                and not self.quiet and not self.inconclusive and self.score < 0.10)

    @property
    def narrows_label(self) -> str:
        """How much a lead would cut down where or what to look at."""
        if not self.detected or self.status != "ok":
            return "-"
        if self.group == "File and metadata" and self.narrows < 0.2:
            return "file history only"
        return "a lot" if self.narrows >= 0.3 else "some" if self.narrows >= 0.15 else "a little"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = plain(self.summary)
        d["score"] = round(self.score, 4)
        return d
