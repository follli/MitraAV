# Writing a plugin probe

A **probe** is one cheap test: does this file contain one identifying signal, and if so, where should the
user look next? MitraAV's own probes and plugin probes work the same way, so anything you can write in
Python (numpy, scipy and Pillow are already installed) can become a row in the results table.

> **A plugin is code that runs with your permissions.** MitraAV never loads one by itself. Files in the
> plugins folder are only *listed* until you enable them; enabling records the file's SHA-256, and a file
> you edit afterwards is switched off again until you re-enable it. Each plugin probe runs in its own
> process (a crash or hang is contained and shown as an error or timeout), but a plugin can still do
> anything your user account can. Read code before enabling it, and don't enable plugins from sources you
> don't trust.

## Quick start

```bash
mitraav-setup --new-plugin                    # copies the example into your plugins folder
mitraav-setup --plugins                       # shows the folder and each file's state
mitraav-setup --enable-plugin loudest_moment.py
mitraav-scan some.wav                         # the new probe is in the results
```

In the app: **Add-ons > Plugin probes...** shows the folder, an *Add example plugin* button and an
Enable/Disable button per file (with a confirmation). Changes apply the next time you open a file. The
default folder is `plugins/` next to the model folder in your user data directory; set
`MITRAAV_PLUGINS` (folders separated by `os.pathsep`) to use another one while developing.

## Anatomy of a plugin

A plugin is one `.py` file. This is the shipped example,
[`src/mitraav/templates/example_plugin.py`](../src/mitraav/templates/example_plugin.py) (it finds the loudest
second of a recording):

```python
from mitraav.plugin import AudioError, Finding, FollowUp, MediaType, Probe, iter_chunks

MITRAAV_API = 1                      # the plugin API version this file was written for

class LoudestMomentProbe(Probe):
    name = "loudest_moment"          # unique, lower case; shown as "loudest moment"
    group = "Sounds"                 # the group the row is listed under
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 120.0                  # seconds before the runner stops it

    def run(self, path):
        ...                          # do the cheap test, then return a Finding
```

Always import from **`mitraav.plugin`**: it is the public, versioned surface. Anything under
`mitraav.core` may change without notice.

### Probe attributes

| Attribute | Meaning |
|---|---|
| `name` | Unique, lower case, `a_b` style. A name already used by a built-in or another plugin is refused. |
| `group` | Group the row is nested under: `Sounds`, `Aircraft and motion`, `Setting and power`, `Text and visuals`, `Visual context`, `File and metadata`, or a new name (new groups sort by their best lead). |
| `media` | A tuple of `MediaType.AUDIO`, `IMAGE`, `VIDEO`, or `MediaType.ANY`. Audio probes also run on a video's soundtrack. |
| `frames_ok` | Image probes only: `True` (default) also runs on five sampled video frames. |
| `timeout` | Seconds. The probe is killed after this and shown as `timeout`. |
| `order` | Lower runs first (default 50). The sound overview is 0 and the scene type 10, so later probes can read their context. |
| `quiet` | `True` for things a person could see for themselves: shown as "context", never counted as a lead. |
| `clutter_sensitive` | `True` if speech or music hides your signal: a miss is then marked inconclusive. |
| `masked_by` | Sound classes that imitate your signal, e.g. `("sirens",)`. |
| `frames_min_agree` | Video frames that must agree (same `meta["label"]`) before a lead stands. |

### What `run(path)` returns: a `Finding`

```python
Finding(
    probe=self.name, detected=True,           # detected=False means "ran, nothing found"
    strength=0.6,                             # 0..1: how clearly the signal is present
    narrows=0.3,                              # 0..1: how much a lead would narrow the search
    summary="Something **measured** that is consistent with ...",
    regions=[dict(label="event near 20 s", t0=20, t1=30, f0=100, f1=800, t_play=18)],
    follow_ups=[FollowUp("A tool", "https://example.org", "what to do there")],
    constrains=["region"],                    # what it helps with: region, date, setting, device ...
    effort=2,                                 # 1 cheap follow-up .. 3 heavy
)
```

- **Result states.** `detected=True` shows a *lead* pill; `detected=False` with `info=True` shows *info*
  ("ran, nothing notable"); `status="not_run"` with a `detail` string means a missing tool or too little
  input (never raise for that); `status="error"` is produced for you if your code raises.
- **Ranking.** Rows sort by `strength * narrows / (1 + 0.25 * (effort - 1))`. A faint lead is flagged
  "very low".
- **Wording.** Wrap key words in `**double asterisks**` (shown bold, stripped in reports). Say
  "consistent with", never "is"; give ranges, not false precision; name what you measured.
- **`regions`** (audio and video): seconds and Hz in file time. A region gives the row a **Spectro**
  button and "play from here" links (`t_play` is where playback starts, default `t0`).
- **`boxes`** (images): `[dict(label=..., x0, y0, x1, y1)]` as fractions of width and height; the
  **Image** button outlines them. Don't put sensitive text in a label.
- **`meta["geo"]`**: a coarse location hint other probes can cross-check, built with `geo_hint(label,
  {"europe"}, weight)` (macro-regions: `europe`, `north_america`, `central_america`, `south_america`,
  `africa`, `asia`, `oceania`). The Summary row flags disagreements between hints.
- **`meta["context"]`**: facts for *later* probes, e.g. `{"scene": {...}}`. Read what earlier probes
  published with `self.context.get("sound")` (`busy` share, `shares` per class) and `self.context.get("scene")`;
  always cope with it being empty.

### Reading the file

```python
from mitraav.plugin import load_audio, iter_chunks, total_seconds
x, sr = load_audio(path)                       # mono float32 and the sample rate (first 5 minutes)
for offset, x, sr in iter_chunks(path):        # long files in 5-minute chunks; offset is in seconds
    ...
```

Non-WAV audio and video soundtracks are decoded with ffmpeg for you. For images use Pillow
(`Image.open(path)`); for a video a probe sees sampled frames as image files. Raise nothing for a file you
can't use: catch `AudioError` and return `status="not_run"` with a clear `detail`.

## Rules of the road

- **One process per run.** Your probe is imported and run in a fresh process each time: no state is kept
  between runs, and top-level code should only import and define things. Do heavy imports inside `run`.
- **Offline.** Scans make no network calls. A plugin that downloads anything breaks that promise; tell the
  user where to look (`FollowUp`) instead of fetching it.
- **Scope.** First-step leads about *places, events and signals*: no face recognition, no tools aimed at
  identifying private individuals, no reading of plates, phone numbers or e-mail addresses.
- **Cheap.** A probe should say whether a signal has potential, not solve the case.
- **Don't edit while enabled.** Any change disables the plugin; run `mitraav-setup --enable-plugin FILE`
  (or use the dialog) again after you've reviewed it.

## Testing your plugin

Run it directly, with no app:

```python
from pathlib import Path
from my_plugin import LoudestMomentProbe
print(LoudestMomentProbe().run(Path("clip.wav")))
```

or through the real runner, which also exercises the process isolation:

```python
from mitraav.core.runner import discover_probes, run_probe
cls = next(c for c in discover_probes() if c.name == "loudest_moment")   # after enabling it
print(run_probe(cls, Path("clip.wav")))
```

`mitraav-scan clip.wav --report out.md` shows how the row reads in a report. The synthetic
[sample generators](../samples/README.md) make test audio with known contents.

## Versioning

`MITRAAV_API = 1` at the top of a plugin declares the API it was written for. A plugin that asks for a
newer API than the installed MitraAV provides is refused with a clear message. Breaking changes to
`mitraav.plugin` will bump the number.
