# MitraAV

A local, first-step OSINT lead finder. Drop an image, audio or video file into the app and press **Start scan**. Many cheap
"probes" each test whether an identifying signal is present, say plainly what they found (or that
they found nothing), and **points you to your next step**.

It is triage: it measures signals and names what it can, offline. It is **not** a solver, it makes
**no network calls during a scan**, and it is not for identifying private individuals (no face
recognition, no plate reading, no phone numbers or e-mail addresses echoed back).

> **Work in progress.** MitraAV is under active development. Probes, results, wording and the
> interface will change as the project develops, and features may be reworked or removed where they
> don't prove useful. Expect rough edges, and treat every result as a lead, never a conclusion.
> Feedback and bug reports are very welcome.

## What it can see and hear

- **Plant, animal and fungus species** in photos and video frames (BioCLIP add-on): candidate species
  with links to their ranges, looked for in the whole picture and in regions of it, and confirmed across
  video frames. People are never reported.
- **Bird species by sound** (BirdNET add-on): species candidates plus a general region and season.
- **What the sounds are:** a sound overview naming trains, horns, bells, sirens, aircraft, animals,
  weather, voices and music, with the weaker guesses kept but muted.
- **Doppler pass-bys:** speed, closest distance and the time offset between events, shown on the signal.
- **Tones and chimes:** church bell pitch, telephone and alert tones, and any other stable pitch measured
  and named (a lift chime, a beeper).
- **Siren patterns** (wail, yelp, two-tone) and **broadband jet pass-bys**.
- **Power and room:** 50/60 Hz mains hum, lighting flicker in video, and reverberation.
- **Text in pictures:** language, script, calling codes, web domains, currencies and units, with the
  text lines outlined (numbers are never echoed).
- **Place type:** the kind of scene (street, shore, woodland, place of worship) as quiet context, plus a
  reverse-search helper for the picture or frame.
- **Location and device tags:** GPS and capture time in photos and video, opened on a map.
- **Hidden data (steganography):** four probes. Appended or extra data in the file structure (after a JPEG or
  PNG ends, private chunks, oversized comments, non-empty free space, attachments and extra tracks); LSB
  replacement in lossless pictures (chi-square and RS analysis, including the transparency channel); JSteg-
  and F5-style changes in JPEGs (stego add-on); LSB data, bits hidden in near-silence and carriers above 15 kHz
  in sound; and unfamiliar user-data SEI payloads, extra tracks and LSB tests on lossless video. Methods built
  to defeat these tests (steghide, OutGuess, LSB matching, adaptive schemes) are not detected, and the
  results say so.
- **How the file was made:** codec and platform fingerprints, JPEG history and edit traces, lossy
  audio, silent gaps and cuts.

Everything is a lead, worded as "consistent with". Numbers are ranges, not facts. Matching against
external datasets (flight tracks, grid frequency logs, incident catalogues) is deliberately left to
the tools the results point to.

## Examples

**A real recording.** A 42-minute field recording scanned in the app (the file is not included). Results
are grouped by what they tell you; the Doppler probe measures speed, closest distance and the time
offset between events, and the bird probe names species and a general region.

![Audio: results grouped by what they tell you](docs/screenshot-results.png)

You don't have to scan everything: the **Run** button on each group header (right edge of the table
above) scans just that group, and for a picture or video frame you can drag a rectangle round one
region and scan only that.

**The identified signal**: the Doppler glide of one fly-by from that recording, with labelled axes and a live time and frequency readout.

![The identified signal, with labelled axes and a live time/frequency readout](docs/screenshot-signal.png)

## Status

Alpha. Developed on Linux; the test suite runs on Python 3.11, 3.12 and 3.13 in CI. Windows and macOS
are untested. Probes are tuned on a small set of real recordings and photos, so expect false positives
and misses: every result is a lead, never a conclusion, and the app says so (confidence,
"inconclusive" for busy audio, "very low" flags).

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 or newer
pip install .                                          # core: audio, video and image probes
mitraav-setup                                          # installs the small add-ons, shows what is missing
```

Optional pieces are **add-ons**. Install them from the **Add-ons** button in the app or with
`mitraav-setup --install <name>`. Nothing is installed unless you ask.

| Add-on | Adds | Download |
|---|---|---|
| `sounds` | names sounds: trains, horns, bells, sirens, aircraft, animals, weather | about 25 MB |
| `scene` | names the kind of place in a picture, as quiet context | about 45 MB |
| `birds` | bird ID by sound, plus a general region and season | about 150 MB |
| `speech` | identifies the spoken language offline (Whisper tiny); the transcript is never shown | about 120 MB |
| `codes` | reads QR codes and product barcodes (web addresses show only the site) | about 3 MB |
| `stego` | JPEG hidden-data tests on DCT coefficients (JSteg- and F5-style), via `jpeglib` | about 80 MB |
| `nature` | species candidates for photos of plants, animals, fungi (BioCLIP) | about 1.7 GB, off by default |

System tools come from your package manager (the app and `mitraav-setup --list` say what is missing):
`ffmpeg` (video and non-WAV audio), `exiftool` (edit traces in images), `tesseract-ocr` (text in
images; add `tesseract-ocr-<lang>` packs for other languages).

## Use

```bash
mitraav [file]                  # the app (drop a file on it, press Start scan; a file passed here scans at once)
mitraav-scan file [--json]      # the same probes from the command line
```

Results are grouped. Each probe shows a **lead** pill (it found something; "lead · very low" when faint) or an **info** pill (it ran and found nothing notable); the detail pane gives its confidence.
For a video or picture, **Scan this frame** (under the player) runs the picture probes on the frame you are looking at; drag a rectangle around a sign or detail to scan just that, enlarged. You can also pick sections to scan one at a time: each group has a **Run** button to run just those probes (the sound overview is rerun first if they need to know how busy the audio is). A lead found under speech or music is kept with a note; a miss there is marked inconclusive. Inactive probes (nothing found, not run) sit greyed at the bottom of their group. "Context" rows
(scene type, reverse-search helper) only report what you could see yourself and are folded away.
Rows that point at a signal in a recording have a small action button: **Spectro** opens the spectrogram on that
finding's signal, **Image** shows the picture full size, with any text lines outlined (the text itself is withheld for numbers) and **Map** opens GPS coordinates in OpenStreetMap. History (next to Open file) lists earlier scans.

The file you dropped stays in view: images show a thumbnail (click to enlarge), videos show a frame
and a small player, and audio gets a player, all above the details. Findings that point at a moment
in the recording offer "▶ play from here" links, so you can listen to what a probe flagged.

## Your own probes

Anyone can add a probe: write a Python file, drop it in the plugins folder and enable it under **Add-ons >
Plugin probes**. Plugins are code that runs with your permissions, so nothing loads until you enable it
and an edited file is switched off again. See [docs/PLUGINS.md](docs/PLUGINS.md) for the guide and API.

## Reports

**Export** (or `mitraav-scan file --report out.md`) saves a Markdown or JSON report: where the clues point
(coarse location hints, cross-checked against each other, with possible disagreements flagged as
questions), the top three leads, other leads, what was inconclusive, and what was checked with nothing
found. The **Summary** row at the top of the results shows the same cross-check.

## Privacy

Scans run on your machine and make no network calls. The only network use is installing an add-on
(you click Install, and the dialog lists exactly what will be downloaded). The History list stores
file paths, names and counts in your user data folder, never results; clear it from the History dialog.
Follow-up links open third-party sites in your browser; MitraAV never uploads your file anywhere.

## Limits worth knowing

- Sound probes are marked **inconclusive** when speech, music or a siren fills the recording, because
  they cannot be trusted there. Quiet bird calls under speech are not heard.
- Species and scene labels come from general models: they name things a person could also see or hear,
  and they are often wrong at the margins. Cultivated plants grow far outside their native ranges.
- Bird range results are a general region lead from a coarse model, not a location.
- Doppler numbers are ranges: the fit models sound-travel delay but assumes straight, level flight at constant speed with a steady tone (on synthetic fly-bys it recovers speed and slant distance to about 1-3%; real recordings are rougher).

## Credits and licences

MitraAV is released under the [MIT License](LICENSE). Third-party models, data and tools it can
download or call have their own terms: see [NOTICE.md](NOTICE.md). In particular the BirdNET models
(an optional add-on) are published for non-commercial use.

## Development

```bash
pip install -e ".[dev]"        # add ,sound ,birds as needed
pytest -q
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture, design rules and how to add a probe.
