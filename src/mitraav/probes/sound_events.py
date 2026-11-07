"""Names the sounds in a recording (trains, bells, sirens, aircraft, animals, weather, ...).

Uses YAMNet, a small AudioSet classifier (521 classes, about 4 MB, Apache-2.0) that runs offline
on the LiteRT runtime. It answers "what is that?": a label with how often and how strongly it was
heard. Labels are leads, not proof (AudioSet classes overlap and distant sounds are often mislabelled),
and the more specific probes (church_bells, siren_pattern, birdsong, doppler_pass) go deeper on the
same sounds.
"""
from __future__ import annotations

import importlib.util
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.models import SETUP_HINT, yamnet_path
from ..core.probe import Probe
from ..core.references import follow_ups

WINDOW = 15600                 # samples at 16 kHz (0.975 s)
HOP = 7800
WIN_SCORE = 0.30
IGNORE = ("silence", "noise", "static", "white noise", "pink noise", "hum")

# (key, display name, label keywords, relevance for locating or identifying a recording)
CATEGORIES = [
    ("rail", "Rail", ("train", "railroad", "rail transport", "subway", "metro", "tram", "locomotive"), True),
    ("bells", "Bells and chimes", ("bell", "chime", "campanology", "carillon", "doorbell", "ding", "glockenspiel",
                                    "mallet percussion", "marimba", "vibraphone", "tuning fork", "music box",
                                    "beep"), True),
    ("sirens", "Sirens and alarms", ("siren", "alarm", "ambulance", "police car", "fire engine",
                                      "civil defense", "buzzer", "smoke detector"), True),
    ("aircraft", "Aircraft", ("aircraft", "airplane", "helicopter", "jet engine", "propeller"), True),
    ("water", "Boats, ships and water", ("boat", "ship", "sailboat", "motorboat", "ocean", "waves",
                                          "surf", "stream", "waterfall", "foghorn", "splash"), True),
    ("road", "Road vehicles", ("vehicle", "car", "truck", "bus", "motorcycle", "traffic", "tire",
                               "skidding", "honking", "toot"), False),
    ("weather", "Weather", ("thunder", "rain", "wind", "storm"), False),
    ("birds", "Birds", ("bird", "crow", "owl", "pigeon", "goose", "duck", "chirp", "tweet", "caw",
                        "gull", "rooster", "chicken", "turkey", "hoot"), True),
    ("animals", "Insects and other animals", ("insect", "cricket", "frog", "cicada", "dog", "cat",
                                               "bark", "howl", "livestock", "cattle", "horse", "sheep",
                                               "pig", "wild animals", "animal", "bee", "fly", "mosquito"), False),
    ("impulsive", "Loud single events", ("gunshot", "gunfire", "explosion", "fireworks", "firecracker",
                                          "bang", "boom", "slam", "knock", "smash", "crash"), False),
    ("voices", "Human voices", ("speech", "conversation", "crowd", "laughter", "shout", "cheering",
                                "applause", "baby", "crying", "whisper", "narration", "yell"), False),
    ("music", "Music", ("music", "guitar", "piano", "drum", "singing", "choir", "organ", "orchestra",
                        "violin", "flute", "trumpet", "accordion", "bagpipes", "synthesizer"), False),
    ("machinery", "Machinery", ("engine", "motor", "drill", "saw", "hammer", "vacuum", "air conditioning",
                                "mechanical fan", "generator", "pump", "tools", "mechanisms"), False),
    ("setting", "Setting", ("outside", "inside"), False),
]


def categorise(label: str) -> str | None:
    low = label.lower()
    if any(low == w or low.startswith(w + ",") for w in IGNORE) or low.startswith("environmental noise"):
        return None
    for key, _, kws, _ in CATEGORIES:
        if any(k in low for k in kws):
            return key
    return None


def aggregate(windows: list[tuple[float, np.ndarray]], labels: list[str]) -> dict:
    """windows: (time, scores[521]). Per label: windows where it ranked top-3 above WIN_SCORE."""
    agg: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen = set()
    for t, s in windows:
        key = round(t / 0.5)                         # overlapping chunks repeat windows
        for k in np.argsort(-s)[:3]:
            if s[k] >= WIN_SCORE and (key, int(k)) not in seen:
                seen.add((key, int(k)))
                agg[labels[k]].append((t, float(s[k])))
    return agg


def summarise(agg: dict) -> list[dict]:
    out = []
    for label, v in agg.items():
        cat = categorise(label)
        if not cat:
            continue
        n, best = len(v), max(s for _, s in v)
        loud = cat == "impulsive"
        if n >= (2 if loud else 3) or best >= 0.6:
            ts = [t for t, _ in v]
            out.append(dict(label=label, cat=cat, n=n, best=best, t0=min(ts), t1=max(ts) + 0.975))
    return sorted(out, key=lambda e: -(e["n"] * e["best"]))


class SoundEventsProbe(Probe):
    name = "sound_events"
    order = 0                 # first, so the other sound probes know how busy the audio is
    group = "Sounds"
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 900.0

    def run(self, path: Path) -> Finding:
        if importlib.util.find_spec("ai_edge_litert") is None:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Sound classification is an add-on: install it from Add-ons in the app, or "
                                  "run: mitraav-setup --install sounds")
        mp = yamnet_path()
        if not mp.exists():
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail=f"Sound classifier model not downloaded (scans never download): {SETUP_HINT}")
        from ai_edge_litert.interpreter import Interpreter
        labels = zipfile.ZipFile(mp).read("yamnet_label_list.txt").decode().splitlines()
        it = Interpreter(model_path=str(mp))
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        windows: list[tuple[float, np.ndarray]] = []
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 1.0:
                    continue
                y = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x.astype(np.float32)
                for a in range(0, len(y) - WINDOW + 1, HOP):
                    it.set_tensor(inp["index"], y[a:a + WINDOW])
                    it.invoke()
                    windows.append((off + a / 16000, it.get_tensor(out["index"])[0].copy()))
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not windows:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 1 s")
        agg = aggregate(windows, labels)
        busy_w = {int(round(t / 0.5)) for label, v in agg.items()
                  if categorise(label) in ("voices", "music") for t, _ in v}
        busy = min(1.0, len(busy_w) / max(len(windows), 1))
        shares: dict[str, float] = {}
        for label, v in agg.items():
            cat = categorise(label)
            if cat:
                shares.setdefault(cat, set()).update(int(round(t / 0.5)) for t, _ in v)
        shares = {c: min(1.0, len(w_) / max(len(windows), 1)) for c, w_ in shares.items()}
        events = summarise(agg)
        if not events:
            return Finding(probe=self.name, detected=False, meta={"busy": busy, "shares": shares},
                           summary="No sound class scored clearly enough to name (generic noise, "
                                   "silence or very distant sounds).")
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            by_cat[e["cat"]].append(e)
        notes, fups, cons, specific = [], [], ["source type"], 0
        for key, name, _, locates in CATEGORIES:
            es = by_cat.get(key)
            if not es:
                continue
            top = es[:3]
            notes.append(f"{name}: " + ", ".join(
                f"**{e['label'].split(',')[0].lower()}** ({e['n']} windows, best {e['best']:.2f}, "
                f"{e['t0']:.0f}-{e['t1']:.0f} s)" for e in top))
            fups += follow_ups("events", key)
            if locates and key != "setting":
                specific += 1
                if "region" not in cons:
                    cons.append("region")
        regions = [dict(label=f"{e['label'].split(',')[0]} ({e['best']:.2f})", t0=max(0.0, e["t0"] - 1),
                        t1=e["t1"] + 1, f0=0.0, f1=8000.0) for e in events[:8]]
        strength = min(1.0, 0.3 + 0.1 * len(by_cat) + 0.1 * specific)
        busy_note = (f" Speech or music fills about **{busy * 100:.0f}%** of the recording, so the "
                     "other sound probes are marked inconclusive." if busy >= 0.35 else "")
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.2, constrains=cons,
            effort=1, regions=regions, follow_ups=fups, meta={"busy": busy, "shares": shares},
            summary="Sounds consistent with: " + "; ".join(notes) + "." + busy_note + " Labels come from a small "
            "general classifier: confusable and distant sounds are often mislabelled, so treat "
            "them as leads and use the specific probes and recordings to confirm.")
