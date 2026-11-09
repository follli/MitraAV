import numpy as np
from scipy.io import wavfile

from mitraav.probes.sharp_events import SharpEventsProbe, describe, find_onsets
from mitraav.probes.time_cues import TimeCuesProbe, hour_cues

SR = 16000


def clicks(times, secs=12, seed=0, decay=0.03, amps=None):
    rng = np.random.default_rng(seed)
    x = 0.003 * rng.standard_normal(SR * secs)
    for k, t in enumerate(times):
        a = int(t * SR)
        n = int(0.1 * SR)
        env = np.exp(-np.arange(n) / (decay * SR))
        x[a:a + n] += (amps[k] if amps else 0.6) * env * rng.standard_normal(n)
    return x


def test_a_regular_series_a_burst_and_an_echo_pair_are_described():
    ev = find_onsets(clicks([2, 3, 4, 5, 6, 7]), SR)
    assert len(ev) == 6 and any("regular series" in n for n in describe(ev))
    burst = find_onsets(clicks([3.0, 3.4, 3.8, 4.2]), SR)
    assert any("burst" in n for n in describe(burst))
    pair = find_onsets(clicks([2.0, 2.5, 5.0, 8.0], amps=[0.6, 0.1, 0.6, 0.6]), SR)
    assert any("quieter" in n and "longer" in n for n in describe(pair))


def test_sharp_events_probe_and_quiet_audio(tmp_path):
    p = tmp_path / "c.wav"
    wavfile.write(p, SR, (clicks([2, 3, 4, 5, 6, 7]) * 30000).astype(np.int16))
    f = SharpEventsProbe().run(p)
    assert f.detected and f.regions and "regular series" in f.summary
    q = tmp_path / "q.wav"
    wavfile.write(q, SR, (0.003 * np.random.default_rng(0).standard_normal(SR * 8) * 30000).astype(np.int16))
    assert not SharpEventsProbe().run(q).detected


def test_clock_strikes_give_an_hour_but_change_ringing_does_not():
    assert "3 o'clock" in hour_cues({"strikes": [10, 14, 18], "distinct": 1})[0]
    assert hour_cues({"strikes": [10, 12, 14, 16, 18, 20], "distinct": 6}) == []
    assert hour_cues({"strikes": [10, 14, 40], "distinct": 1}) == [] or "2 o'clock" in hour_cues({"strikes": [10, 14, 40], "distinct": 1})[0]


def test_time_cues_read_the_shared_context(tmp_path):
    probe = TimeCuesProbe()
    probe.context = {"sound": {"labels": {"rooster": 3, "cricket": 8}, "shares": {"birds": 0.5}},
                     "bells": {"strikes": [5, 9, 13, 17], "distinct": 1}}
    f = probe.run(tmp_path / "x.wav")
    assert f.detected and "4 o'clock" in f.summary and "rooster" in f.summary and "dawn" in f.summary and "insects" in f.summary
    empty = TimeCuesProbe()
    empty.context = {}
    assert not empty.run(tmp_path / "x.wav").detected
