# Architecture and contributing

MitraAV is a local, first-step lead finder. Many cheap **probes** each test whether one identifying
signal is present in an image, audio or video file, say so in calibrated language, and point to
public tools for follow-up. It is triage, not a solver.

## Design rules

- **Scope.** Provide a signal and whatever can be said cheaply and offline. Matching against external
  datasets (flight tracks, grid-frequency logs, incident catalogues) is out of scope; where an
  external tool is needed, point to it.
- **Name what a sound is** when that can be done cheaply and offline (train, siren, bell...), at least
  "yes, that is a train horn", more precisely where possible.
- **Calibrated wording.** "Consistent with", never "is". Ranges, not false precision.
- **Guardrails.** No face recognition, no plate reading, no tools aimed at identifying private
  individuals. Phone numbers, e-mail addresses and plate-like strings are never echoed back.
- **No network during a scan.** The only network use is installing an add-on on request.
- **Missing optional dependency or tool means `status="not_run"`**, never a crash. A hung probe is
  killed by the runner (`status="timeout"`).

## Layout

```
src/mitraav/
  core/      finding.py (Finding, MediaType, plain()), probe.py (Probe ABC), runner.py (discovery,
             per-probe process, scan, media detection), audio.py (load, chunking, silence skipping),
             spectro.py, lines.py, tracks.py, frames.py, models.py, addons.py, history.py,
             birdworker.py, birdgeo.py, nature.py, references.py
  probes/    one file per probe, auto-discovered
  references/ follow-up links and lookup tables as YAML, label files as text (package data)
  gui/       main_window.py, preview.py (thumbnail / audio / video player), spectro_view.py,
             addons_dialog.py, history_dialog.py, theme.py
  cli.py  setup_models.py
tests/  scripts/  testmedia/  samples/make_samples.py
```

## Adding a probe

One file in `probes/` subclassing `Probe` (`name`, `group`, `media`, `timeout`, `run(path) -> Finding`),
entries in a `references/*.yaml` file for follow-ups, and tests. Report `strength` (how clearly the
signal is present), `narrows` (how much a lead would cut down where or what to look at), `constrains`,
`effort`, follow-ups and, where useful, `regions` (time/frequency boxes the signal viewer opens on).
Regions may carry `t_play` (seconds, file time), where the GUI's "play from here" link starts
playback (a few seconds before the event); otherwise it uses `t0`.
Wrap key words in `**double asterisks**` in summaries; the GUI shows them bold and `plain()` strips
them for the CLI and JSON.

Useful class attributes: `quiet` (reports only what a person could see; shown as context, not a lead),
`order` (lower runs first), `clutter_sensitive` (marked inconclusive when speech or music fills the
recording), `masked_by` (sound-event categories that can imitate the signal, e.g. a siren),
`frames_ok` (image probes also run on sampled video frames).

## How a scan works

`runner.planned(path)` lists the probes for a file type; `scan()` runs each in its own spawned
process (so a hang can be killed) and announces each as it starts and finishes. `sound_events` (shown as "sound overview") runs
first and publishes how busy the audio is and which sound categories are present; the runner uses
that to mark other sound probes inconclusive. Audio longer than five minutes is processed in
overlapping chunks, with leading digital silence skipped; file-absolute times are reported.

`tones_chimes` is one probe made of two analyses (`signal_tones.py`: table match plus measured pitches; `bells.py`:
tuned-bell strikes); those classes are helpers, not probes. Opening a file only lists the probes; the
Start scan button (or `start_scan()` from the command line) runs them.

## Shared context

A probe can publish what it learned in `Finding.meta["context"]` (small dicts keyed by topic: `sound`,
`scene`, `text`; see `core/context.py`). The runner merges these and passes the result to every later
probe as `self.context`, and rebuilds it from earlier findings for a partial (per-group) run. Probes only
read it and must work without it. In use today: the sound overview's busy share and sound classes
(masking of other sound probes), and the scene model's whole-picture result, which `nature_id` reuses.
`scene_type` runs early (`order = 10`) so it is available. `image_text` publishes language and country
hints for later use.

## Scoring shown to users

`Finding.score = strength x narrows / (1 + 0.25 x (effort - 1))` orders results. The table shows
only the result pills (lead, lead · very low, info, inconclusive, context); the detail pane gives the
confidence word (clear / likely / faint). Groups keep scan order while probes
report in and sort by best lead once the scan is done; inactive probes sink to the bottom of a group. A faint lead is flagged
"very low"; results from busy audio are "inconclusive"; probes that ran and found
nothing notable show "info" (blue), which, like every other row, can be clicked for details.

## Add-ons

Optional probes live behind add-ons (`core/addons.py`): `sounds`, `scene`, `birds`, `nature`. Each has
a status and install steps run in a subprocess; the Add-ons dialog and `mitraav-setup` share them.
Models are downloaded once into the user data folder; scans only check that they exist.

## Testing

`pytest -q` runs everything; tests that need an add-on model, ffmpeg or downloaded media skip
themselves when it is absent. Most probes are tested on synthetic signals. `testmedia/` has a manifest
of freely licensed real recordings, photos and video (see its README) and `scripts/fetch_testmedia.py`
downloads them politely; the media itself is never committed. Do not commit third-party or personal
media.

## Known limits

- Probes are tuned on a small set of real material; expect false positives and misses.
- The spectrogram dialog computes on the GUI thread (up to 240 s at a time).
- The Doppler tracer's ridge search is pure Python and slow on very long clips.
- Hum and pass-by analyse only the first five minutes of sound.
- Doppler numbers come from a simplified model (no sound-travel delay in the curve shape).
- Windows and macOS are untested.
