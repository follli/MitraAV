import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from mitraav.core import datatones as dt
from mitraav.probes.data_tones import DataTonesProbe

SR = 16000


def tone(f, d, a=0.3):
    t = np.arange(int(d * SR)) / SR
    return a * np.sin(2 * np.pi * f * t)


def silence(d):
    return np.zeros(int(d * SR))


def noisy(x, a=0.01, seed=0):
    return x + a * np.random.default_rng(seed).standard_normal(len(x))


def dtmf_seq(keys="5551234"):
    parts = [silence(0.3)]
    for k in keys:
        i = dt.DTMF_KEYS.index(k)
        parts += [tone(dt.DTMF_ROWS[i // 4], 0.08) + tone(dt.DTMF_COLS[i % 4], 0.08), silence(0.06)]
    return np.concatenate(parts + [silence(0.3)])


def morse_sig(text="CQ CQ DE W1AW", wpm=20, f=700):
    inv = {v: k for k, v in dt.MORSE.items()}
    u = 1.2 / wpm
    parts = [silence(0.3)]
    for word in text.split():
        for li, ch in enumerate(word):
            for ei, e in enumerate(inv[ch]):
                parts.append(tone(f, u * (3 if e == "-" else 1)))
                if ei < len(inv[ch]) - 1:
                    parts.append(silence(u))
            if li < len(word) - 1:
                parts.append(silence(3 * u))
        parts.append(silence(7 * u))
    return np.concatenate(parts + [silence(0.5)])


def sstv_header(vis=44):
    parts = [silence(0.2), tone(1900, 0.3), tone(1200, 0.01), tone(1900, 0.3), tone(1200, 0.03)]
    bits = [(vis >> i) & 1 for i in range(7)]
    for b in bits + [sum(bits) % 2]:
        parts.append(tone(1100 if b else 1300, 0.03))
    return np.concatenate(parts + [tone(1200, 0.03), silence(0.3)])


def afsk(nbits=1200, seed=0):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    return np.concatenate([silence(0.2)] + [tone(1200 if b else 2200, 1 / 1200) for b in bits] + [silence(0.2)])


def test_dtmf_key_presses_are_counted_but_digits_are_not_returned():
    keys = dt.dtmf(noisy(dtmf_seq()), SR)
    assert len(keys) == 7 and all(k["key"] is None for k in keys)


def test_morse_is_decoded_when_clean():
    m = dt.morse(noisy(morse_sig()), SR)
    assert m and m["text"] == "CQ CQ DE W1AW" and 12 <= m["wpm"] <= 22 and abs(m["tone"] - 700) <= 25


def test_sstv_header_and_vis_mode():
    h = dt.sstv(noisy(sstv_header()), SR)
    assert len(h) == 1 and h[0]["mode"] == "Martin M1"


def test_afsk_1200_is_found_and_steady_tones_and_noise_are_not():
    assert dt.afsk1200(noisy(afsk()), SR)
    t = np.arange(SR * 5) / SR
    assert not dt.afsk1200(0.3 * np.sin(2 * np.pi * 1200 * t), SR)
    n = 0.1 * np.random.default_rng(1).standard_normal(SR * 8)
    assert not dt.afsk1200(n, SR) and not dt.dtmf(n, SR) and dt.morse(n, SR) is None and not dt.sstv(n, SR)


def test_probe_reports_each_signal_and_stays_quiet_on_music_like_tones(tmp_path):
    x = noisy(np.concatenate([dtmf_seq(), morse_sig(), sstv_header(), afsk()]))
    p = tmp_path / "all.wav"
    wavfile.write(p, SR, (x * 20000).astype(np.int16))
    f = DataTonesProbe().run(p)
    assert f.detected and "DTMF" in f.summary and "Morse" in f.summary and "SSTV" in f.summary and "AFSK" in f.summary
    assert "5551234" not in f.summary and f.regions and f.follow_ups
    t = np.arange(SR * 6) / SR
    music = 0.2 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0))
    q = tmp_path / "music.wav"
    wavfile.write(q, SR, (music * 20000).astype(np.int16))
    assert not DataTonesProbe().run(q).detected
