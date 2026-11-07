"""Siren and horn patterns: measures the pitch pattern of a loud, narrow tone.

Two kinds are recognised: alternating two-tone (hi-lo) and a periodic sweep (wail or yelp). The
output is the measured pitches and timing, which is what national siren and horn standards are
described in; it does not say which country. Emergency sirens differ by country, vehicle type and
era, so use the numbers with the reference links.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import spectrogram

from ..core.audio import AudioError, iter_chunks
from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups

FLO, FHI = 300.0, 2200.0
MIN_PROM_DB = 18.0
MIN_SEG_S = 4.0


def pitch_series(x: np.ndarray, sr: int):
    """Dominant narrow peak per frame: (times, freq, valid) with a ~32 ms hop."""
    n = int(2 ** round(np.log2(sr * 0.064)))
    f, t, s = spectrogram(x, fs=sr, nperseg=n, noverlap=n // 2, mode="psd")
    m = (f >= FLO) & (f <= FHI)
    db = 10 * np.log10(s[m] + 1e-14)
    fm = f[m]
    i = db.argmax(axis=0)
    peak = db[i, np.arange(db.shape[1])]
    prom = peak - np.median(db, axis=0)
    # sharpness: the peak must stand above its own neighbourhood a few bins away
    near = np.array([np.median(db[max(0, j - 12):j + 13, c]) for c, j in enumerate(i)])
    valid = (prom >= MIN_PROM_DB) & (peak - near >= 15.0)
    return t, fm[i], valid


def _segments(valid: np.ndarray, hop: float, gap: float = 0.25, min_s: float = MIN_SEG_S):
    segs, start, last = [], None, None
    for k in np.flatnonzero(valid):
        if start is None:
            start = last = k
        elif (k - last) * hop <= gap:
            last = k
        else:
            if (last - start + 1) * hop >= min_s:
                segs.append((start, last))
            start = last = k
    if start is not None and (last - start + 1) * hop >= min_s:
        segs.append((start, last))
    return segs


def two_tone(f: np.ndarray, valid: np.ndarray, hop: float) -> dict | None:
    """Alternation between two stable pitches, as run lengths."""
    idx = np.flatnonzero(valid)
    if len(idx) < 20:
        return None
    runs, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= 3 and abs(np.log(f[k] / np.median(f[cur[-5:]]))) < 0.04:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    runs = [r for r in runs if len(r) * hop >= 0.25]
    best = None
    chain = []
    for r in runs:
        fr = float(np.median(f[r]))
        if chain and (r[0] - chain[-1]["end"]) * hop <= 0.5 and abs(np.log(fr / chain[-1]["f"])) > 0.08 \
                and (len(chain) < 2 or abs(np.log(fr / chain[-2]["f"])) < 0.04):
            chain.append(dict(f=fr, start=r[0], end=r[-1]))
        else:
            if len(chain) >= 6 and (best is None or len(chain) > len(best)):
                best = chain
            chain = [dict(f=fr, start=r[0], end=r[-1])]
    if len(chain) >= 6 and (best is None or len(chain) > len(best)):
        best = chain
    if not best:
        return None
    lens = np.array([(c["end"] - c["start"] + 1) * hop for c in best])
    if np.std(lens) / np.mean(lens) > 0.45:
        return None
    hi = float(np.median([c["f"] for c in best[0::2]]))
    lo = float(np.median([c["f"] for c in best[1::2]]))
    return dict(kind="two-tone", hi=max(hi, lo), lo=min(hi, lo), period=float(2 * np.mean(lens)),
                t0=best[0]["start"] * hop, t1=best[-1]["end"] * hop, cycles=len(best) // 2)


def sweep(f: np.ndarray, valid: np.ndarray, hop: float, seg: tuple[int, int]) -> dict | None:
    a, b = seg
    y = np.log(f[a:b + 1])
    ok = valid[a:b + 1]
    y = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    ratio = float(np.exp(np.percentile(y, 95) - np.percentile(y, 5)))
    if ratio < 1.25:
        return None
    y = y - y.mean()
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    ac /= ac[0] + 1e-12
    hi_lag = min(len(ac) - 1, int(14 / hop))
    neg = np.flatnonzero(ac[:hi_lag] < 0)
    if not len(neg):
        return None
    start = int(neg[0])                                  # past the first zero crossing
    if start + 2 >= hi_lag:
        return None
    j = int(np.argmax(ac[start:hi_lag])) + start
    if ac[j] < 0.5 or j * hop < 0.15:
        return None
    return dict(kind="sweep", lo=float(np.exp(np.percentile(y + np.log(f[a:b + 1]).mean(), 5))),
                hi=float(np.exp(np.percentile(y + np.log(f[a:b + 1]).mean(), 95))),
                period=float(j * hop), ratio=ratio, t0=a * hop, t1=b * hop, strength=float(ac[j]))


def analyse(x: np.ndarray, sr: int, offset: float) -> list[dict]:
    t, fr, valid = pitch_series(x, sr)
    if len(t) < 40:
        return []
    hop = float(t[1] - t[0])
    out = []
    tt = two_tone(fr, valid, hop)
    if tt:
        out.append(tt)
    for seg in _segments(valid, hop, min_s=6.0):
        if tt and seg[0] * hop <= tt["t1"] and seg[1] * hop >= tt["t0"]:
            continue                                        # a hi-lo siren is not also a wail
        sw = sweep(fr, valid, hop, seg)
        if sw:
            out.append(sw)
    for e in out:
        e["t0"] += offset
        e["t1"] += offset
    return out


class SirenProbe(Probe):
    name = "siren_pattern"
    group = "Sounds"
    clutter_sensitive = True
    media = (MediaType.AUDIO, MediaType.VIDEO)
    timeout = 300.0

    def run(self, path: Path) -> Finding:
        found: list[dict] = []
        n = 0
        try:
            for off, x, sr in iter_chunks(path):
                if len(x) / sr < 8 or sr < 4000:
                    continue
                n += 1
                found += analyse(x, sr, off)
        except AudioError as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=str(e))
        if not n:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="No usable audio: need non-silent audio of at least 8 s")
        uniq: list[dict] = []
        for e in sorted(found, key=lambda e: e["t0"]):          # overlapping chunks repeat events
            if not any(abs(e["t0"] - u["t0"]) < 3 and e["kind"] == u["kind"] for u in uniq):
                uniq.append(e)
        if not uniq:
            return Finding(probe=self.name, detected=False,
                           summary="No two-tone or sweeping siren pattern found.")
        notes = []
        for e in uniq[:4]:
            if e["kind"] == "two-tone":
                notes.append(f"**alternating two-tone** pattern near {e['t0']:.0f}-{e['t1']:.0f} s: "
                             f"about **{e['hi']:.0f} and {e['lo']:.0f} Hz**, a full cycle every "
                             f"{e['period']:.2f} s ({e['cycles']} cycles), consistent with a "
                             "hi-lo emergency siren or horn")
            else:
                kind = "yelp" if e["period"] < 1.5 else "wail"
                notes.append(f"periodic **pitch sweep** near {e['t0']:.0f}-{e['t1']:.0f} s: about "
                             f"{e['lo']:.0f} to {e['hi']:.0f} Hz, repeating every {e['period']:.1f} s, "
                             f"consistent with a **{kind}**-type siren (or an alarm with a sweeping tone)")
        regions = [dict(label=f"{e['kind']} siren near {e['t0']:.0f} s", t0=max(0.0, e["t0"] - 2),
                        t1=e["t1"] + 2, f0=0.7 * e["lo"], f1=1.5 * e["hi"]) for e in uniq[:4]]
        return Finding(probe=self.name, detected=True, strength=min(1.0, 0.45 + 0.1 * len(uniq)),
                       narrows=0.2, constrains=["region", "source type"], effort=2, regions=regions,
                       follow_ups=follow_ups("audio", "sirens"),
                       summary="; ".join(notes) + ". Siren sounds differ by country, vehicle type and "
                       "era: compare the pitches and timing with the references, and note a moving "
                       "siren's pitch also shifts with Doppler.")
