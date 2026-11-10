Doppler pass-by detector, one file.

Needs: Python 3.10+, numpy, scipy   (pip install numpy scipy)
Optional: ffmpeg on your PATH, only for audio that is not a WAV file, and for video.

Run:
    python doppler_standalone.py recording.wav
    python doppler_standalone.py recording.mp3 --json

What it does: looks for a smooth downward glide in the pitch of an aircraft or engine tone and fits a
Doppler model, giving the closest approach time, speed and closest slant distance. The numbers are
ranges that assume straight, level flight at constant speed and a steady tone. It cannot identify the
aircraft: match speed, distance and time against ADS-B tracks (links are printed).

Good input: a recording with a clear propeller, rotor or engine tone, at least 5 seconds of the pass.
Jets are broadband and often show no clean tone. Speech, music or sirens can hide or imitate a pass-by,
and this standalone version does not check for them. Very long files are scanned in 5-minute pieces
(first 3 hours). It makes no network calls.

Generated from the MitraAV sources by scripts/build_standalone_doppler.py.
