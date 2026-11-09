"""Text in images: script, language and country clues from signs, shop fronts and screens.

Runs Tesseract (a system tool) and reads clues from the recognised text: a phone number's country
code, a web address's country domain, currency symbols, speed and temperature units, and which
language the words look like. Only the clue is reported. Phone numbers, e-mail addresses and
number-plate-like strings are never repeated in the output (this tool is not for identifying
individuals), and clues are hints: businesses and global services use foreign codes and domains.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from ..core.finding import Finding, MediaType
from ..core.geo import hint, macros_for_phrase
from ..core.probe import Probe
from ..core.references import _load, follow_ups

MIN_CONF = 60
MAX_SIDE = 3000
# Function words of three or more letters only: one- and two-letter words ("a", "o", "de") match
# OCR noise by chance, which once made foliage "Portuguese".
STOPWORDS = {
    "English": {"the", "and", "for", "with", "from", "are", "you", "open", "street", "not", "this", "that"},
    "Spanish": {"los", "las", "del", "por", "con", "para", "una", "que", "calle", "abierto", "este", "como"},
    "French": {"les", "des", "pour", "avec", "une", "dans", "rue", "est", "ouvert", "sur", "pas", "vous"},
    "German": {"der", "die", "das", "und", "von", "mit", "für", "den", "ist", "nicht", "straße", "strasse"},
    "Italian": {"per", "con", "che", "del", "della", "nel", "via", "una", "aperto", "sono", "non", "gli"},
    "Portuguese": {"com", "para", "uma", "por", "rua", "aberto", "dos", "das", "que", "não", "nos", "mais"},
    "Dutch": {"het", "een", "van", "voor", "met", "straat", "niet", "dat", "zijn", "ook", "naar", "bij"},
}
MIN_TEXT_WORDS = 6        # real words (3+ letters) needed before language or word lists are reported
MIN_TEXT_SHARE = 0.45     # ...and they must be a good share of what OCR returned (noise is mostly symbols)
PLATE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9-]{5,9}$")
PHONE = re.compile(r"(?<![\w+])\+\s?(\d{1,3})((?:[\s().-]*\d){6,})")
DOMAIN = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.([a-z]{2})(?:\.[a-z]{2})?(?=[/\s,;:]|$)", re.IGNORECASE)


def scales_for(size: tuple[int, int]) -> list[float]:
    """Scales to try. Tesseract is picky about letter height, so small pictures are also tried
    enlarged and very large ones reduced; the best result wins."""
    big = max(size)
    out = [min(1.0, MAX_SIDE / big)]
    if big <= 2200:
        out.append(min(2.0, MAX_SIDE / big))
    if big >= 2500:
        out.append(0.5 * min(1.0, MAX_SIDE / big))
    return out


def _tesseract(exe: str, img: Image.Image, use: str, workdir: Path):
    tmp = workdir / f"in{img.width}.png"
    img.save(tmp)
    r = subprocess.run([exe, str(tmp), "stdout", "-l", use, "--psm", "11", "tsv"],
                       capture_output=True, text=True, timeout=120)
    words = []
    rows: dict[tuple, list] = {}
    for line in r.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 12 and parts[11].strip():
            try:
                conf = int(float(parts[10]))
                x, y, w, h = (int(parts[i]) for i in (6, 7, 8, 9))
            except ValueError:
                continue
            words.append((parts[11].strip(), conf))
            if conf >= MIN_CONF:
                rows.setdefault(tuple(parts[2:5]), []).append((parts[11].strip(), conf, x, y, x + w, y + h))
    lines = []
    for ws in rows.values():                                    # one box per text line, as fractions
        lines.append(dict(text=" ".join(w[0] for w in ws), conf=sum(w[1] for w in ws) / len(ws),
                          x0=min(w[2] for w in ws) / img.width, y0=min(w[3] for w in ws) / img.height,
                          x1=max(w[4] for w in ws) / img.width, y1=max(w[5] for w in ws) / img.height))
    return words, tmp, lines


def _iou(a: dict, b: dict) -> float:
    iw = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    ih = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)


def line_boxes(lines: list[dict], limit: int = 25) -> list[dict]:
    """Viewer boxes for text lines. Lines with digits, "@" or plate-like strings are boxed but their
    text is withheld, like everywhere else in the app. Duplicates from several scales are merged."""
    kept: list[dict] = []
    for ln in sorted(lines, key=lambda d: -d["conf"]):
        toks = ln["text"].split()
        if not any(t.isalpha() and len(t) >= 3 for t in toks) and not re.search(r"\d", ln["text"]):
            continue                                            # stray marks
        if any(_iou(ln, k) > 0.4 for k in kept):
            continue
        kept.append(ln)
    out = []
    for ln in kept[:limit]:
        sensitive = re.search(r"[\d@]", ln["text"]) or any(PLATE.match(re.sub(r"^\W+|\W+$", "", t))
                                                           for t in ln["text"].split())
        label = "number or code (text withheld)" if sensitive else ln["text"][:48]
        out.append(dict(label=label, **{k: round(ln[k], 4) for k in ("x0", "y0", "x1", "y1")}))
    return sorted(out, key=lambda b: (b["y0"], b["x0"]))


def _real_words(words: list[tuple[str, int]]) -> int:
    return sum(1 for w, c in words if c >= MIN_CONF and w.isalpha() and len(w) >= 3)


def ocr(path: Path) -> tuple[list[tuple[str, int]], str] | None:
    """Words with confidence, and the raw OSD script line (may be empty). None if no tesseract."""
    got = ocr_full(path)
    return None if got is None else got[:2]


def ocr_full(path: Path):
    """Like ocr(), plus the text lines with their boxes (fractions of the picture)."""
    exe = shutil.which("tesseract")
    if not exe:
        return None
    langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True).stdout.split()[1:]
    use = "+".join(sorted(lang for lang in langs if lang not in ("osd", "snum"))) or "eng"
    workdir = Path(tempfile.mkdtemp())
    with Image.open(path) as im:
        gray = im.convert("L")
    best = None
    merged: list[tuple[str, int]] = []
    all_lines: list[dict] = []
    for scale in scales_for(gray.size):
        img = gray if abs(scale - 1.0) < 1e-9 else gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))))
        got = _tesseract(exe, img, use, workdir)
        merged += got[0]                                        # different scales read different lines
        all_lines += got[2]
        if best is None or _real_words(got[0]) > _real_words(best[0]):
            best = got
    words, tmp = merged, best[1]
    script = ""
    if "osd" in langs:
        o = subprocess.run([exe, str(tmp), "stdout", "--psm", "0"], capture_output=True, text=True,
                           timeout=60)
        m = re.search(r"Script:\s*(\w+)", o.stdout)
        conf = re.search(r"Script confidence:\s*([\d.]+)", o.stdout)
        # OSD is unreliable on little text: need a confident score and enough words
        enough = sum(1 for _, c in words if c >= MIN_CONF) >= 6
        script = m[1] if m and conf and float(conf[1]) >= 8.0 and enough else ""
    return words, script, all_lines


def plausible_phone(code_and_rest: str) -> bool:
    """E.164 numbers have 8-15 digits; evenly spaced runs are axis labels, not a phone number."""
    digits = re.sub(r"\D", "", code_and_rest)
    if not 8 <= len(digits) <= 15:
        return False
    groups = [int(g) for g in re.findall(r"\d+", code_and_rest)]
    if len(groups) >= 4 and len({b - a for a, b in zip(groups, groups[1:])}) == 1:
        return False
    return True


def clues(words: list[tuple[str, int]]) -> dict:
    good = [w for w, c in words if c >= MIN_CONF]
    text = " ".join(good)
    tables = _load("codes")
    out = {"phone": [], "domain": [], "currency": [], "units": [], "language": None,
           "plates": 0, "words": []}
    for m in PHONE.finditer(text):
        if not plausible_phone(m[1] + m[2]):
            continue
        for n in (3, 2, 1):                                   # longest calling code that matches
            code = m[1][:n]
            if code in tables["calling_codes"] and len(m[1]) >= n:
                if (code, tables["calling_codes"][code]) not in out["phone"]:
                    out["phone"].append((code, tables["calling_codes"][code]))
                break
    for m in DOMAIN.finditer(text):
        cc = m[1].lower()
        if cc in tables["cctlds"] and cc not in [c for c, _ in out["domain"]]:
            out["domain"].append((cc, tables["cctlds"][cc]))
    for sym, where in tables["currencies"].items():
        if sym in text and (sym, where) not in out["currency"]:
            out["currency"].append((sym, where))
    for u, where in tables["units"].items():
        if re.search(rf"(?<![\w]){re.escape(u)}(?![\w])", text, re.IGNORECASE if u.isalpha() else 0):
            out["units"].append((u, where))
    toks = [re.sub(r"^\W+|\W+$", "", w) for w in good]
    alpha = [t for t in toks if t.isalpha() and len(t) >= 3]
    out["real_text"] = len(alpha) >= MIN_TEXT_WORDS and len(alpha) >= MIN_TEXT_SHARE * max(len(toks), 1)
    out["plates"] = sum(1 for t in toks if PLATE.match(t))
    if out["real_text"]:
        low = [t.lower() for t in alpha]
        scores = {lang: sum(1 for t in low if t in sw) for lang, sw in STOPWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] >= 4 and scores[best] >= 2 * sorted(scores.values())[-2]:
            out["language"] = best
        out["words"] = list(dict.fromkeys(t for t in alpha if not PLATE.match(t)))[:20]   # once each
    return out


class ImageTextProbe(Probe):
    name = "image_text"
    group = "Text and visuals"
    media = (MediaType.IMAGE,)
    timeout = 240.0

    def run(self, path: Path) -> Finding:
        res = ocr_full(path)
        if res is None:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Reading text needs Tesseract: sudo apt install tesseract-ocr "
                                  "(add tesseract-ocr-<language> packs to read other languages)")
        words, script, lines = res
        c = clues(words)
        good = [w for w, conf in words if conf >= MIN_CONF]
        c_pre = c
        if len(good) < 2 or not (c_pre["real_text"] or c_pre["phone"] or c_pre["domain"]
                                 or c_pre["currency"] or c_pre["units"]):
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No readable text found (too small, blurred, stylised or not "
                                   "horizontal; foliage and other texture can fool OCR, which is "
                                   "ignored).")
        notes, cons = [], []
        if c["language"]:
            notes.append(f"the words look like **{c['language']}** (common function words)")
            cons += ["language", "region"]
        if script and script != "Latin":
            notes.append(f"the script looks like **{script}**; install the matching Tesseract language "
                         "pack to read it")
            cons += ["language", "region"]
        for code, where in c["phone"]:
            notes.append(f"a phone number with the **+{code}** country code appears (the number is "
                         f"not shown), consistent with {where}")
            cons.append("region")
        for cc, where in c["domain"]:
            notes.append(f"a web address ending in **.{cc}** appears, consistent with {where}")
            cons.append("region")
        for sym, where in c["currency"]:
            notes.append(f"the currency mark **{sym}** appears, consistent with {where}")
            cons.append("region")
        for u, where in c["units"]:
            notes.append(f"the unit **'{u}'** appears, consistent with {where}")
            cons.append("region")
        if c["words"]:
            notes.append("readable words include: " + ", ".join(c["words"][:12]))
        if c["plates"]:
            notes.append(f"{c['plates']} string(s) look like number plates and are not shown")
        specific = len(c["phone"]) + len(c["domain"]) + len(c["currency"]) + len(c["units"]) \
            + (1 if c["language"] else 0) + (1 if script and script != "Latin" else 0)
        if not specific:
            return Finding(probe=self.name, detected=False,
                           summary="Text was read but nothing in it points to a language or country. "
                                   + notes[-1] if notes else "Text was read but nothing in it points "
                                   "to a language or country.")
        regions = sorted({where for key in ("phone", "domain", "currency", "units") for _, where in c[key]})
        return Finding(probe=self.name, detected=True, strength=min(1.0, 0.10 + 0.15 * specific),
                       meta={"context": {"text": dict(language=c["language"], script=script or None,
                                                      regions=regions)},
                             "geo": hint("text clues: " + ", ".join(regions[:3]),
                                         set().union(*(macros_for_phrase(r) for r in regions)) if regions else set(),
                                         min(0.35, 0.15 * len(regions)))},
                       narrows=0.3, constrains=sorted(set(cons)), effort=1, boxes=line_boxes(lines),
                       follow_ups=follow_ups("metadata", "text"),
                       summary="; ".join(notes) + ". Hints only: global services use foreign codes and currencies.")
