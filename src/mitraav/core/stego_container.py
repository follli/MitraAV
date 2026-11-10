"""File-structure checks for hidden data: what a file holds *outside* the picture, sound or video.

Parsers for JPEG, PNG, GIF, WAV, MP3 (ID3) and MP4/MOV that find data after the logical end of the
file, unusual or oversized metadata, bad checksums and free space that is not empty. Pure Python; the
file is read, never decoded. Many findings have innocent causes (camera padding, editors, thumbnails),
so each observation carries a severity and a plain-language reason.
"""
from __future__ import annotations

import mmap
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOOL_WORDS = re.compile(rb"(steghide|openstego|outguess|jsteg|f5 stego|stegosuite|openpuff|deepsound|silenteye|"
                        rb"camouflage|snow stego|stegano|invisible secrets|steganography)", re.IGNORECASE)
SIGNATURES = [
    (b"PK\x03\x04", "a ZIP archive"), (b"Rar!\x1a\x07", "a RAR archive"), (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"\x1f\x8b\x08", "gzip data"), (b"BZh", "bzip2 data"), (b"\xfd7zXZ\x00", "xz data"),
    (b"%PDF", "a PDF"), (b"\x7fELF", "an ELF program"), (b"MZ", "a Windows program"),
    (b"Salted__", "OpenSSL-encrypted data"), (b"-----BEGIN PGP", "PGP data"), (b"SQLite format 3", "an SQLite database"),
    (b"\xff\xd8\xff", "another JPEG image"), (b"\x89PNG\r\n\x1a\n", "another PNG image"), (b"GIF8", "a GIF image"),
    (b"OggS", "an Ogg stream"), (b"RIFF", "a RIFF (WAV/AVI) file"), (b"ID3", "an MP3 audio file"),
]
BENIGN_TRAILING = 3                  # bytes of newline/padding some cameras and editors add


@dataclass
class Observation:
    kind: str                         # short machine label, e.g. "trailing_data"
    severity: float                   # 0..1: how unusual (not proof of anything)
    text: str                         # plain-language description (calibrated)
    offset: int = 0
    size: int = 0


def entropy(b: bytes) -> float:
    """Shannon entropy in bits per byte (8 = random or compressed, text is about 4-5)."""
    if not b:
        return 0.0
    c = np.bincount(np.frombuffer(b, np.uint8), minlength=256).astype(float)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def signature_of(b: bytes) -> str | None:
    for magic, name in SIGNATURES:
        if b.startswith(magic):
            return name
    if b[257:262] == b"ustar":
        return "a tar archive"
    return None


def trailing(data: bytes, end: int, what: str) -> Observation | None:
    """Bytes after the logical end of the file."""
    n = max(len(data) - end, 0)
    if n <= BENIGN_TRAILING:
        return None
    tail = data[end:end + 1_048_576]           # a sample is enough; the file may be gigabytes
    sig = signature_of(tail.lstrip(b"\x00"))
    ent = entropy(tail[:65536])
    sev = 0.25 + (0.25 if n >= 1024 else 0.0) + (0.3 if sig else 0.0) + (0.15 if ent >= 7.2 and n >= 256 else 0.0)
    bits = [f"{n:,} bytes follow the end of the {what}"]
    if sig:
        bits.append(f"they begin like {sig}")
    if ent >= 7.2 and n >= 256:
        bits.append(f"they look random or compressed ({ent:.1f} bits per byte)")
    elif n >= 256 and ent < 5.5:
        bits.append(f"they look like text or structured data ({ent:.1f} bits per byte)")
    return Observation("trailing_data", min(sev, 0.95), ", ".join(bits), end, n)


def _tool_words(blob: bytes, where: str, offset: int = 0) -> Observation | None:
    m = TOOL_WORDS.search(blob)
    if not m:
        return None
    return Observation("tool_name", 0.5, f"{where} mentions '{m.group(1).decode('latin-1')}', the name of a "
                       "hiding tool (or text about one)", offset + m.start(), len(m.group(0)))


# --- JPEG -----------------------------------------------------------------------------------------
JPEG_KNOWN_APP = {0xE0: "JFIF", 0xE1: "Exif/XMP", 0xE2: "ICC/MPF", 0xED: "Photoshop", 0xEE: "Adobe"}


def inspect_jpeg(data: bytes) -> list[Observation]:
    obs: list[Observation] = []
    if not data.startswith(b"\xff\xd8"):
        return obs
    pos, n, mpf = 2, len(data), False
    end = None
    while pos < n - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xD9:
            end = pos + 2
            break
        if marker == 0x00 or 0xD0 <= marker <= 0xD8 or marker == 0x01:
            pos += 2
            continue
        if pos + 4 > n:
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg = data[pos + 4:pos + 2 + length]
        if marker == 0xFE:                               # comment
            text = seg.rstrip(b"\x00")
            if len(text) >= 200:
                obs.append(Observation("comment", 0.3 + (0.2 if len(text) >= 2000 else 0.0),
                                       f"a {len(text):,}-byte comment segment (comments are normally short)",
                                       pos, len(text)))
            tool = _tool_words(text, "A comment segment", pos)
            if tool:
                obs.append(tool)
        elif 0xE0 <= marker <= 0xEF:
            if marker == 0xE2 and seg.startswith(b"MPF\x00"):
                mpf = True
            if marker not in JPEG_KNOWN_APP and (len(seg) >= 4096 or (len(seg) >= 512 and entropy(seg) >= 7.2)):
                obs.append(Observation("odd_segment", 0.35, f"an unusual application segment (APP{marker - 0xE0}) of "
                                       f"{len(seg):,} bytes", pos, len(seg)))
            tool = _tool_words(seg, f"The APP{marker - 0xE0} segment", pos)
            if tool:
                obs.append(tool)
        if marker == 0xDA:                               # start of scan: skip the entropy-coded data
            pos += 2 + length
            while pos < n - 1:
                if data[pos] == 0xFF and data[pos + 1] not in (0x00, 0xFF) and not 0xD0 <= data[pos + 1] <= 0xD7:
                    break
                pos += 1
            continue
        pos += 2 + length
    if end is None:
        obs.append(Observation("truncated", 0.2, "the JPEG has no end marker (truncated or damaged)", n, 0))
        return obs
    tail = data[end:]
    if mpf and tail.startswith(b"\xff\xd8"):
        return obs                                       # multi-picture file (phones): extra images are expected
    t = trailing(data, end, "JPEG image")
    if t:
        obs.append(t)
    return obs


# --- PNG ------------------------------------------------------------------------------------------
PNG_KNOWN = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"cHRM", b"gAMA", b"iCCP", b"sBIT", b"sRGB", b"bKGD", b"hIST",
             b"tRNS", b"pHYs", b"sPLT", b"tIME", b"iTXt", b"tEXt", b"zTXt", b"eXIf", b"acTL", b"fcTL", b"fdAT",
             b"cICP", b"mDCV", b"cLLI", b"oFFs", b"pCAL", b"sCAL", b"gIFg", b"gIFx", b"sTER", b"iDOT", b"prVW",
             b"mkBF", b"mkTS", b"mkBS", b"mkBT", b"vpAg", b"caNv", b"orNT", b"dSIG", b"caBX"}


def inspect_png(data: bytes) -> list[Observation]:
    obs: list[Observation] = []
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return obs
    pos, n, end = 8, len(data), None
    bad_crc = 0
    while pos + 12 <= n:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:pos + 12 + length]
        if len(crc) < 4 or length > n:
            obs.append(Observation("truncated", 0.2, "a PNG chunk runs past the end of the file", pos, 0))
            break
        if struct.unpack(">I", crc)[0] != (zlib.crc32(ctype + body) & 0xFFFFFFFF):
            bad_crc += 1
        name = ctype.decode("latin-1")
        if ctype not in PNG_KNOWN:
            critical = ctype[0:1].isupper()
            obs.append(Observation("unknown_chunk", 0.5 if critical else 0.25 + (0.2 if length >= 1024 else 0.0),
                                   f"an unrecognised {'critical ' if critical else ''}chunk '{name}' of {length:,} bytes",
                                   pos, length))
        if ctype in (b"tEXt", b"zTXt", b"iTXt") and length >= 1024:
            keyword = body.split(b"\x00", 1)[0][:30].decode("latin-1")
            obs.append(Observation("big_text", 0.3 + (0.2 if length >= 20000 else 0.0),
                                   f"a {length:,}-byte text chunk ('{keyword}')", pos, length))
        tool = _tool_words(body[:4096], f"The '{name}' chunk", pos) if ctype in (b"tEXt", b"iTXt", b"zTXt") else None
        if tool:
            obs.append(tool)
        pos += 12 + length
        if ctype == b"IEND":
            end = pos
            break
    if bad_crc:
        obs.append(Observation("bad_crc", 0.4, f"{bad_crc} chunk checksum(s) fail (the file was edited, or "
                               "written by a tool that did not fix them)", 0, bad_crc))
    if end is not None:
        t = trailing(data, end, "PNG image")
        if t:
            obs.append(t)
    return obs


# --- GIF ------------------------------------------------------------------------------------------
def inspect_gif(data: bytes) -> list[Observation]:
    if not data.startswith((b"GIF87a", b"GIF89a")):
        return []
    end = data.rfind(b"\x3b")
    obs: list[Observation] = []
    if end >= 0:
        t = trailing(data, end + 1, "GIF image")
        if t:
            obs.append(t)
    for m in re.finditer(rb"\x21\xfe", data):                  # comment extension: sub-blocks follow
        i, total = m.end(), 0
        while i < len(data) and data[i] != 0:
            total += data[i]
            i += 1 + data[i]
        if total >= 200:
            obs.append(Observation("comment", 0.35, f"a {total:,}-byte comment (GIF comments are normally short)", m.start(), total))
    return obs


# --- WAV ------------------------------------------------------------------------------------------
WAV_KNOWN = {b"fmt ", b"data", b"fact", b"LIST", b"bext", b"cue ", b"smpl", b"inst", b"JUNK", b"junk", b"PAD ",
             b"iXML", b"axml", b"ds64", b"id3 ", b"ID3 ", b"minf", b"elm1", b"_PMX", b"cart", b"levl", b"peak",
             b"PEAK", b"afsp", b"plst", b"labl", b"note", b"ltxt", b"disp", b"umid", b"SNDM", b"regn", b"strc"}


def inspect_wav(data: bytes) -> list[Observation]:
    if not (data.startswith(b"RIFF") and data[8:12] == b"WAVE"):
        return []
    obs: list[Observation] = []
    declared = struct.unpack("<I", data[4:8])[0] + 8
    pos = 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + (size if cid != b"data" else 0)]   # never copy the audio itself
        if cid not in WAV_KNOWN and size >= 64:
            obs.append(Observation("unknown_chunk", 0.4 + (0.1 if size >= 4096 else 0.0),
                                   f"an unrecognised chunk '{cid.decode('latin-1')}' of {size:,} bytes", pos, size))
        if cid in (b"JUNK", b"junk", b"PAD ") and size >= 512 and any(body):
            obs.append(Observation("filler_not_empty", 0.6, f"a {size:,}-byte filler chunk that is not empty "
                                   "(filler is normally zeros)", pos, size))
        tool = _tool_words(body[:8192], f"The '{cid.decode('latin-1')}' chunk", pos) if cid != b"data" else None
        if tool:
            obs.append(tool)
        nxt = pos + 8 + size + (size & 1)
        if nxt <= pos:
            break
        pos = nxt
    if declared < len(data):
        t = trailing(data, declared + (declared & 1), "WAV file")
        if t:
            obs.append(t)
    elif declared > len(data) + 1:
        obs.append(Observation("truncated", 0.1, "the WAV header declares more data than the file holds", len(data), 0))
    return obs


# --- MP3 (ID3) --------------------------------------------------------------------------------------
def inspect_mp3(data: bytes) -> list[Observation]:
    obs: list[Observation] = []
    if not data.startswith(b"ID3") or len(data) < 10:
        return obs
    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
    tag = data[10:10 + size]
    pos = 0
    while pos + 10 <= len(tag):
        fid, fsize = tag[pos:pos + 4], struct.unpack(">I", tag[pos + 4:pos + 8])[0] if data[3] == 3 else \
            (tag[pos + 4] << 21) | (tag[pos + 5] << 14) | (tag[pos + 6] << 7) | tag[pos + 7]
        if not fid.isalnum() or fsize <= 0:
            break
        if fid in (b"GEOB", b"PRIV") and fsize >= 512:
            obs.append(Observation("big_frame", 0.4, f"an ID3 '{fid.decode()}' frame of {fsize:,} bytes (arbitrary "
                                   "data container)", 10 + pos, fsize))
        elif fid == b"APIC" and fsize >= 1_000_000:
            obs.append(Observation("big_frame", 0.25, f"embedded cover art of {fsize:,} bytes", 10 + pos, fsize))
        tool = _tool_words(tag[pos + 10:pos + 10 + min(fsize, 8192)], f"The ID3 '{fid.decode()}' frame", 10 + pos)
        if tool:
            obs.append(tool)
        pos += 10 + fsize
    padding = tag[pos:]
    if len(padding) >= 1024 and any(padding[:65536]):
        obs.append(Observation("padding_not_empty", 0.5, f"{len(padding):,} bytes of tag padding that are not empty "
                               "(padding is normally zeros)", 10 + pos, len(padding)))
    return obs


# --- MP4 / MOV ------------------------------------------------------------------------------------
MP4_KNOWN = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"meta", b"moof", b"mfra", b"sidx", b"styp",
             b"pdin", b"prft", b"emsg", b"ssix", b"uuid", b"udta", b"junk", b"pnot", b"cmov", b"iods"}
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, btype = struct.unpack(">I4s", data[pos:pos + 8])
        hdr = 8
        if size == 1 and pos + 16 <= end:
            size, hdr = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end + 0:
            yield pos, btype, max(size, hdr), hdr, False
            return
        yield pos, btype, size, hdr, True
        pos += size


def inspect_mp4(data: bytes) -> list[Observation]:
    if data[4:8] not in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
        return []
    obs: list[Observation] = []
    end_of_boxes = 0
    for pos, btype, size, hdr, ok in _boxes(data, 0, len(data)):
        name = btype.decode("latin-1")
        if not ok:
            if pos == 0:
                obs.append(Observation("bad_box", 0.3, f"box '{name}' declares a size that runs past the end of the file", pos, size))
                end_of_boxes = None
            else:
                end_of_boxes = pos                       # what follows the last good box is not boxes: stray bytes
            break
        end_of_boxes = pos + size
        body = data[pos + hdr:pos + size]
        if btype not in MP4_KNOWN:
            obs.append(Observation("unknown_box", 0.4 + (0.1 if size >= 4096 else 0.0),
                                   f"an unrecognised top-level box '{name}' of {size:,} bytes", pos, size))
        if btype in (b"free", b"skip", b"wide", b"junk") and size >= 1024 and any(body[:262144]):
            obs.append(Observation("filler_not_empty", 0.6, f"a {size:,}-byte '{name}' (free space) box that is not "
                                   "empty (free space is normally zeros)", pos, size))
        if btype == b"uuid" and body[:16] != XMP_UUID and size >= 1024:
            obs.append(Observation("uuid_box", 0.35, f"a {size:,}-byte 'uuid' box with an unfamiliar identifier "
                                   f"({body[:16].hex()})", pos, size))
        if btype in (b"moov", b"udta", b"meta"):
            tool = _tool_words(body[:1_000_000], f"The '{name}' box", pos)
            if tool:
                obs.append(tool)
        if btype == b"moov":
            for cpos, ctype, csize, chdr, cok in _boxes(data, pos + hdr, pos + size):
                if ctype == b"udta" and csize >= 65536:
                    obs.append(Observation("big_udta", 0.4, f"a {csize:,}-byte user-data box inside the movie box", cpos, csize))
    if end_of_boxes and end_of_boxes < len(data):
        t = trailing(data, end_of_boxes, "last box")
        if t:
            obs.append(t)
    return obs


class _Mapped(mmap.mmap):
    """Read-only memory map with the bytes methods the parsers use, so a multi-GB file is never copied into memory."""

    def startswith(self, prefix) -> bool:
        if isinstance(prefix, tuple):
            return any(self.startswith(p) for p in prefix)
        return self[:len(prefix)] == prefix


LOAD_LIMIT = 64 * 1024 * 1024


def inspect_file(path: Path) -> list[Observation]:
    """Run the parser that matches the file's own signature (never trusting the extension alone).

    Files over 64 MB are memory-mapped instead of read whole (a 3-hour WAV is 2.6 GB)."""
    p = Path(path)
    size = p.stat().st_size
    if size == 0:
        return []
    if size <= LOAD_LIMIT:
        return _inspect(p.read_bytes())
    with open(p, "rb") as fh:
        mm = _Mapped(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return _inspect(mm)
        finally:
            mm.close()


def _inspect(data) -> list[Observation]:
    head = bytes(data[:16])
    if head.startswith(b"\xff\xd8"):
        return inspect_jpeg(data)
    if head.startswith(b"\x89PNG"):
        return inspect_png(data)
    if head.startswith(b"GIF8"):
        return inspect_gif(data)
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return inspect_wav(data)
    if head.startswith(b"ID3"):
        return inspect_mp3(data)
    if head[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
        return inspect_mp4(data)
    return []
