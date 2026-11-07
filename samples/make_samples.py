"""Synthetic fly-bys closer to a real security-camera recording than a clean sweep."""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

C, SR, NFFT = 343.0, 22050, 4096
ENBW = 1.5 * SR / NFFT

PRESETS = {  # v m/s, h altitude above mic m, d ground offset m, tc emission-time CPA s, snr dB
    "prop_far": dict(secs=110, events=[
        dict(v=75, h=600, d=1500, f0=85, nh=10, tc=55, snr=13)]),
    "prop_near": dict(secs=50, events=[
        dict(v=60, h=150, d=250, f0=110, nh=8, tc=25, snr=22)]),
    "two_events": dict(secs=150, events=[
        dict(v=80, h=500, d=1200, f0=95, nh=9, tc=35, snr=17),
        dict(v=95, h=700, d=1800, f0=105, nh=9, tc=110, snr=16)]),
}

JETS = {"jet_pass": dict(secs=100, v=200, h=1500, d=2000, tc=50, snr=8)}


def colored_noise(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec /= np.sqrt(np.maximum(f, 40.0) / 40.0)       # pink-ish: more energy at low frequencies
    x = np.fft.irfft(spec, n)
    return x / x.std()


def gate(t, a, b):
    return 0.5 * (np.tanh((t - a) / 0.5) - np.tanh((t - b) / 0.5))


def synth(p, seed=0):
    rng = np.random.default_rng(seed)
    secs = p["secs"]
    n = SR * secs
    t = np.arange(n) / SR
    noise = 0.02 * colored_noise(n, rng)
    fw, psd = welch(noise, fs=SR, nperseg=NFFT)

    def amp(f0, snr_db):   # tone amplitude giving snr_db above local noise in one FFT bin
        band = (fw >= 0.7 * f0) & (fw <= 1.4 * f0)
        return np.sqrt(2 * ENBW * np.median(psd[band]) * 10 ** (snr_db / 10))

    x = noise.copy()
    for f0, snr, a, b in [(60, 20, 0, 1e9), (120, 17, 0, 1e9), (180, 12, 0, 1e9),
                          (310, 14, 0.2 * secs, 1e9), (470, 12, 0, 0.6 * secs)]:
        x += amp(f0, snr) * gate(t, a, b) * np.sin(2 * np.pi * f0 * t + rng.random() * 6.28)
    wob = 190 + 35 * np.sin(2 * np.pi * 0.5 * t) + 15 * np.sin(2 * np.pi * 1.3 * t)
    x += amp(190, 18) * gate(t, 0.3 * secs, 0.3 * secs + 8) * np.sin(2 * np.pi * np.cumsum(wob) / SR)

    for ev in p["events"]:
        te = np.arange(-60, secs + 60, 0.01)                       # emission times
        r = np.sqrt((ev["v"] * (te - ev["tc"])) ** 2 + ev["d"] ** 2 + (ev["h"] - 1.5) ** 2)
        ta = te + r / C                                            # arrival times
        tem, rr = np.interp(t, ta, te), np.interp(t, ta, r)        # when emitted, range now
        rmin = float(np.hypot(ev["d"], ev["h"] - 1.5))
        gain = (rmin / rr) / (1 + (rr / 6000) ** 4)                # 1/r, inaudible past ~6 km
        a0 = amp(ev["f0"], ev["snr"])
        y = np.zeros(n)
        for k in range(1, ev["nh"] + 1):
            if k * ev["f0"] > 0.45 * SR:
                break
            y += (a0 / k ** 1.2) * np.exp(-(k - 1) * rr / 8000) * np.sin(
                2 * np.pi * k * ev["f0"] * tem + rng.random() * 6.28)
        x += y * gain
        print(f"  event: true speed {ev['v']} m/s, slant distance {rmin:.0f} m, "
              f"heard CPA near {ev['tc'] + rmin / C:.0f} s")
    return x * (0.5 / np.abs(x).max())


def synth_jet(p, seed=1):
    """Broadband pass-by: band-shaped noise, Doppler-scaled by resampling at emission time."""
    rng = np.random.default_rng(seed)
    secs, n = p["secs"], SR * p["secs"]
    t = np.arange(n) / SR
    bg = 0.02 * colored_noise(n, rng)
    src = np.fft.rfft(rng.standard_normal(n + 60 * SR))
    f = np.fft.rfftfreq(n + 60 * SR, 1 / SR)
    src *= 1 / (1 + (f / 900.0) ** 2)                  # jet-like: energy rolls off above ~1 kHz
    src = np.fft.irfft(src)
    src /= src.std()
    te = np.arange(-30, secs + 30, 0.01)
    r = np.sqrt((p["v"] * (te - p["tc"])) ** 2 + p["d"] ** 2 + (p["h"] - 1.5) ** 2)
    ta = te + r / C
    tem, rr = np.interp(t, ta, te), np.interp(t, ta, r)
    rmin = float(np.hypot(p["d"], p["h"] - 1.5))
    y = np.interp(tem + 30, np.arange(src.size) / SR, src) * (rmin / rr)
    x = bg + 0.02 * 10 ** (p["snr"] / 20) * y
    print(f"  jet: true speed {p['v']} m/s, slant distance {rmin:.0f} m, "
          f"distance/speed {rmin / p['v']:.1f} s, heard CPA near {p['tc'] + rmin / C:.0f} s")
    return x * (0.5 / np.abs(x).max())


if __name__ == "__main__":
    out = Path(__file__).parent
    for name, p in PRESETS.items():
        print(name)
        wavfile.write(out / f"{name}.wav", SR, (synth(p) * 32767).astype(np.int16))
    for name, p in JETS.items():
        print(name)
        wavfile.write(out / f"{name}.wav", SR, (synth_jet(p) * 32767).astype(np.int16))
