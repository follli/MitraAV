# Samples

`python samples/make_samples.py` generates synthetic fly-by recordings here (`prop_far.wav`,
`prop_near.wav`, `two_events.wav`, `jet_pass.wav`) and prints the ground truth for each (speed,
closest distance, time of closest approach). They are not committed: generate them when you want
to try the app on known material. For real recordings with known origins see `../testmedia/`.

`python samples/make_demo.py` writes `demo_everything.wav`, `.mp4` and `.jpg`: synthetic media that
lights up most probes at once (hum, a Doppler fly-by, bells, a dial tone, a siren, a jet pass-by,
reverberation and a lossy-codec cliff in the audio; flicker, a location tag and encoder traces in the
video; GPS, camera and editor tags plus a sign with a phone code, web address and units in the
image), and prints the ground truth. Handy for trying the app, for screenshots, and as a regression test.
