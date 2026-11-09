"""Hidden data in video: extra tracks, stray bytes in the codec's own data units, and (for lossless
codecs only) LSB replacement in the frames.

* **User-data SEI messages** in H.264 and H.265 streams are a documented place to carry arbitrary bytes
  next to the picture. Encoders write a few of their own (x264 settings, captions, HDR metadata); an
  unfamiliar identifier or a lot of high-entropy payload is worth a look.
* **Lossless codecs** (FFV1, HuffYUV, UT Video, raw): the same chi-square and RS pixel tests as for
  pictures are run on sampled frames. In lossy codecs the pixel LSBs are rewritten by the encoder, so
  tools that decode a lossy video and test its pixels (a common "video steganalysis" shortcut) mostly
  measure the codec; this probe does not.

Container-level extras (attachments, data tracks, appended bytes, free-space boxes) are reported by the
hidden-data container probe. Hiding in motion vectors or DCT coefficients of H.264 is not detected.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.finding import Finding, MediaType
from ..core.probe import Probe
from ..core.references import follow_ups
from ..core.stego_container import entropy
from .stego_image import analyse_array, describe

LOSSLESS_VIDEO = ("ffv1", "huffyuv", "ffvhuff", "utvideo", "rawvideo", "magicyuv", "png", "qtrle", "cfhd_lossless")
X264_UUID = bytes.fromhex("dc45e9bde6d948b7962cd820d923eeef")
MAX_STREAM = 24_000_000          # bytes of elementary stream read for the SEI scan


def _probe(path: Path) -> dict | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-print_format", "json",
                            "-show_streams", str(path)], capture_output=True, text=True, timeout=60)
        return (json.loads(r.stdout or "{}").get("streams") or [None])[0]
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _unescape(b: bytes) -> bytes:
    return b.replace(b"\x00\x00\x03", b"\x00\x00")


def sei_user_data(stream: bytes, hevc: bool) -> dict[bytes, list[int]]:
    """{uuid: [payload sizes]} for user_data_unregistered SEI messages in an Annex B stream."""
    out: dict[bytes, list[int]] = defaultdict(list)
    pos, n = 0, len(stream)
    starts = []
    while True:
        i = stream.find(b"\x00\x00\x01", pos)
        if i < 0:
            break
        starts.append(i + 3)
        pos = i + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        if s >= n:
            continue
        if hevc:
            nal_type, hdr = (stream[s] >> 1) & 0x3F, 2
            is_sei = nal_type in (39, 40)
        else:
            nal_type, hdr = stream[s] & 0x1F, 1
            is_sei = nal_type == 6
        if not is_sei:
            continue
        body = _unescape(stream[s + hdr:e])
        i = 0
        while i + 2 < len(body):
            ptype = 0
            while i < len(body) and body[i] == 0xFF:
                ptype += 255
                i += 1
            if i >= len(body):
                break
            ptype += body[i]
            i += 1
            psize = 0
            while i < len(body) and body[i] == 0xFF:
                psize += 255
                i += 1
            if i >= len(body):
                break
            psize += body[i]
            i += 1
            payload = body[i:i + psize]
            if ptype == 5 and len(payload) >= 16:
                out[payload[:16]].append(len(payload) - 16)
            i += psize
            if i < len(body) and body[i] == 0x80:      # rbsp trailing bits
                break
    return out


def read_sei(path: Path, codec: str) -> dict[bytes, list[int]]:
    exe = shutil.which("ffmpeg")
    if not exe or codec not in ("h264", "hevc"):
        return {}
    bsf, fmt = ("h264_mp4toannexb", "h264") if codec == "h264" else ("hevc_mp4toannexb", "hevc")
    try:
        p = subprocess.Popen([exe, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-bsf:v", bsf,
                              "-f", fmt, "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = p.stdout.read(MAX_STREAM)
        p.kill()
        p.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return sei_user_data(data, codec == "hevc")


def lossless_frames(path: Path, info: dict, count: int = 6) -> list[np.ndarray]:
    """Sampled frames in the codec's own colour model: RGB streams as RGB, YUV streams as the luma
    plane. (Converting YUV to RGB would invent new low bits and fake the statistics.)"""
    exe = shutil.which("ffmpeg")
    w, h = int(info.get("width", 0)), int(info.get("height", 0))
    if not exe or not w or not h or w * h > 8_000_000:
        return []
    pix = str(info.get("pix_fmt", ""))
    rgb = pix.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "bgra", "rgba", "0rgb", "rgb0", "bgr0", "0bgr"))
    if not rgb and not pix.startswith(("yuv", "gray", "nv")):
        return []
    fmt, ch = ("rgb24", 3) if rgb else ("gray", 1)
    try:
        p = subprocess.run([exe, "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", str(count),
                            "-f", "rawvideo", "-pix_fmt", fmt, "pipe:1"], capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return []
    size = w * h * ch
    raw = p.stdout
    shape = (h, w, 3) if ch == 3 else (h, w)
    return [np.frombuffer(raw[i * size:(i + 1) * size], np.uint8).reshape(shape) for i in range(len(raw) // size)]


class StegoVideoProbe(Probe):
    name = "hidden_data_video"
    group = "Hidden data"
    media = (MediaType.VIDEO,)
    timeout = 600.0

    def run(self, path: Path) -> Finding:
        info = _probe(path)
        if not info:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="ffprobe is needed (and a video track) for the video hidden-data checks")
        codec = info.get("codec_name", "")
        notes: list[str] = []
        strength = 0.0
        sei = read_sei(path, codec)
        for uuid, sizes in sei.items():
            if uuid == X264_UUID:
                continue
            total = sum(sizes)
            if total >= 256 or len(sizes) >= 20:
                notes.append(f"**{len(sizes)}** user-data SEI message(s) with the unfamiliar identifier {uuid.hex()} "
                             f"carry {total:,} bytes in total (encoders write their own settings here, but it can hold "
                             "any bytes)")
                strength = max(strength, 0.45 + (0.15 if total >= 4096 else 0.0))
        pixel_note = ""
        if codec in LOSSLESS_VIDEO:
            frames = lossless_frames(path, info)
            if frames:
                hits = []
                for fr in frames:
                    res = analyse_array(fr)
                    bullets, s = describe(res, fr.size)
                    if bullets:
                        hits.append((bullets[0], s))
                if len(hits) >= max(1, len(frames) // 2):
                    notes.append(f"{len(hits)} of {len(frames)} sampled frames show LSB-replacement signs: {hits[0][0]}")
                    strength = max(strength, max(s for _, s in hits))
        else:
            pixel_note = (f" Pixel LSB tests were not run: {codec or 'this'} is a lossy codec that rewrites the lowest "
                          "bits, so they would mostly measure the encoder.")
        if not notes:
            return Finding(
                probe=self.name, detected=False, info=True,
                summary="No unfamiliar user-data SEI payload" + (" or LSB-replacement sign in the lossless frames"
                                                                 if codec in LOSSLESS_VIDEO else "") + " found."
                + pixel_note + " Data hidden in motion vectors or DCT coefficients is not tested, so this is not "
                               "proof that nothing is hidden.")
        return Finding(
            probe=self.name, detected=True, strength=strength, narrows=0.4, effort=2,
            constrains=["hidden content"], follow_ups=follow_ups("stego", "video"),
            summary=" • ".join(notes) + " • Worth a look with ffprobe; the container probe lists extra tracks.")


_ = entropy        # re-exported for tests that build synthetic SEI payloads
