"""Provenance and AI-generation clues in the metadata of any file.

Looks only at metadata blocks (EXIF, XMP, PNG text chunks, C2PA/JUMBF manifests, container tags), never
at pixels. C2PA "Content Credentials" are *detected*, not verified: checking a signature needs the
manifest's certificates and a trust list, so the result points to a verifier. AI tool names and
prompts in metadata are trivially removed or forged, so their absence proves nothing.
"""
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .stego_container import _boxes

AI_TOOLS = ("Midjourney", "DALL-E", "DALL·E", "Stable Diffusion", "StableDiffusion", "Adobe Firefly", "Firefly",
            "Imagen", "Gemini", "Flux", "ComfyUI", "AUTOMATIC1111", "NovelAI", "Leonardo.Ai", "Ideogram", "Sora",
            "Runway", "Kling", "DreamStudio", "InvokeAI", "Craiyon", "Bing Image Creator", "Copilot Designer",
            "Veo", "Hailuo", "Luma Dream Machine", "SDXL", "Stability AI", "OpenAI")
EDITORS = ("Photoshop", "Lightroom", "GIMP", "Snapseed", "Pixelmator", "Affinity", "Canva", "Capture One",
           "Premiere", "Final Cut", "DaVinci", "CapCut", "Filmora", "iMovie", "Kdenlive")
PNG_AI_KEYS = {"parameters": "Stable Diffusion (AUTOMATIC1111-style)", "prompt": "ComfyUI-style", "workflow": "ComfyUI",
               "invokeai_metadata": "InvokeAI", "sd-metadata": "InvokeAI (older)", "Dream": "InvokeAI (older)",
               "generation_data": "an image generator"}
C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")


@dataclass
class Provenance:
    c2pa: bool = False
    c2pa_where: str = ""
    generators: list[str] = field(default_factory=list)       # names found in/near the manifest or XMP
    source_types: list[str] = field(default_factory=list)     # IPTC digital source types
    ai_tools: list[str] = field(default_factory=list)
    editors: list[str] = field(default_factory=list)
    prompt: str = ""
    gen_params: dict[str, str] = field(default_factory=dict)
    gen_source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def declares_ai(self) -> bool:
        return bool(self.ai_tools or self.gen_source or any("trainedAlgorithmic" in s or "algorithmicMedia" in s
                                                           for s in self.source_types))


def _blobs(data: bytes) -> list[tuple[str, bytes]]:
    """Metadata-carrying pieces of a file: (where, bytes)."""
    out: list[tuple[str, bytes]] = []
    if data.startswith(b"\xff\xd8"):
        pos, n = 2, len(data)
        while pos + 4 <= n and data[pos] == 0xFF:
            m = data[pos + 1]
            if m == 0xDA or m == 0xD9:
                break
            length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            if 0xE0 <= m <= 0xEF or m == 0xFE:
                out.append((f"JPEG {'COM' if m == 0xFE else f'APP{m - 0xE0}'}", data[pos + 4:pos + 2 + length]))
            pos += 2 + length
    elif data.startswith(b"\x89PNG"):
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            if ctype not in (b"IDAT", b"fdAT"):
                out.append((f"PNG {ctype.decode('latin-1')}", data[pos + 8:pos + 8 + length]))
            pos += 12 + length
            if ctype == b"IEND":
                break
    elif data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
        for pos, btype, size, hdr, ok in _boxes(data, 0, len(data)):
            if ok and btype in (b"moov", b"uuid", b"meta", b"udta", b"jumb", b"XMP_"):
                out.append((f"MP4 {btype.decode('latin-1')}", data[pos + hdr:pos + min(size, 4_000_000)]))
    elif data.startswith(b"RIFF"):
        out.append(("RIFF", data[12:200_000]))
    elif data.startswith(b"ID3"):
        out.append(("ID3", data[:200_000]))
    else:
        out.append(("start of file", data[:65536]))
    return out


def _png_text(blob_name: str, blob: bytes) -> tuple[str, str] | None:
    kind = blob_name[4:]
    try:
        if kind == "tEXt":
            k, _, v = blob.partition(b"\x00")
            return k.decode("latin-1"), v.decode("latin-1", "replace")
        if kind == "zTXt":
            k, _, rest = blob.partition(b"\x00")
            return k.decode("latin-1"), zlib.decompress(rest[1:]).decode("utf-8", "replace")
        if kind == "iTXt":
            k, _, rest = blob.partition(b"\x00")
            comp, rest = rest[0], rest[2:]
            _lang, _, rest = rest.partition(b"\x00")
            _tkey, _, text = rest.partition(b"\x00")
            return k.decode("utf-8", "replace"), (zlib.decompress(text) if comp else text).decode("utf-8", "replace")
    except (zlib.error, IndexError, ValueError):
        return None
    return None


def _find_names(text: str, names) -> list[str]:
    return [n for n in names if re.search(r"(?<![A-Za-z])" + re.escape(n) + r"(?![a-z])", text)]


def analyse(path: Path) -> Provenance:
    p = Path(path)
    data = p.read_bytes()
    if len(data) > 32_000_000:
        data = data[:24_000_000] + data[-4_000_000:]
    res = Provenance()
    for where, blob in _blobs(data):
        is_c2pa = (b"c2pa" in blob and b"jumb" in blob) or where in ("PNG caBX",) or blob[:16] == C2PA_UUID
        text = blob.decode("latin-1")
        if is_c2pa:
            res.c2pa, res.c2pa_where = True, res.c2pa_where or where
            for m in re.finditer(r"(?:claim_generator(?:_info)?|softwareAgent|name)\W{1,8}([A-Za-z0-9][A-Za-z0-9 ._/\-]{2,60})",
                                 text):
                g = m.group(1).strip()
                if g and g not in res.generators and not g.lower().startswith(("c2pa", "http")):
                    res.generators.append(g)
        for m in re.finditer(r"digitalsourcetype/(\w+)", text, re.IGNORECASE):
            if m.group(1) not in res.source_types:
                res.source_types.append(m.group(1))
        for m in re.finditer(r"DigitalSourceType[^<>\"]{0,10}[>\"=]\s*([^<>\"]{3,200})", text):
            tail = re.sub(r"^.*/", "", m.group(1).strip())
            if tail and tail not in res.source_types:
                res.source_types.append(tail)
        if where.startswith("PNG t") or where.startswith("PNG z") or where.startswith("PNG i"):
            kv = _png_text(where, blob)
            if kv:
                key, val = kv
                if key in PNG_AI_KEYS and not res.gen_source:
                    res.gen_source = PNG_AI_KEYS[key]
                    if key == "parameters":
                        first = val.split("Negative prompt:")[0].split("\nSteps:")[0].strip()
                        res.prompt = first[:160]
                        for k in ("Steps", "Sampler", "CFG scale", "Seed", "Size", "Model"):
                            mm = re.search(rf"{k}:\s*([^,\n]+)", val)
                            if mm:
                                res.gen_params[k] = mm.group(1).strip()
                    elif key in ("prompt", "workflow") and val.lstrip().startswith("{"):
                        res.notes.append("a ComfyUI node graph is stored in the file")
                elif key == "Comment" and '"prompt"' in val and not res.gen_source:
                    res.gen_source = "NovelAI-style"
                    mm = re.search(r'"prompt"\s*:\s*"([^"]{0,160})', val)
                    res.prompt = mm.group(1) if mm else ""
                elif key == "Software":
                    text += " " + val
        res.ai_tools += [t for t in _find_names(text, AI_TOOLS) if t not in res.ai_tools]
        res.editors += [t for t in _find_names(text, EDITORS) if t not in res.editors]
    res.generators = [g for g in res.generators if len(g) > 2][:6]
    return res
