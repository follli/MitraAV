"""QR codes and barcodes in a picture or video frame, read offline (zxing-cpp, the `codes` add-on).

What a code holds can be a strong clue: a QR code's web address, a Wi-Fi network, a map location, or a
product barcode whose prefix names the country where the brand owner registered it. The probe reports the
kind of code and, for web addresses, only the site (the rest can carry a personal token). Contact cards,
phone numbers, e-mail addresses and Wi-Fi passwords are never shown. Nothing is opened or fetched.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from ..core.finding import Finding, FollowUp, MediaType
from ..core.geo import hint, macros_for_latlon, macros_for_phrase
from ..core.probe import Probe
from ..core.references import _load

GEO_URI = re.compile(r"^geo:(-?\d{1,2}\.\d+),(-?\d{1,3}\.\d+)", re.IGNORECASE)


def gs1_country(code: str) -> str | None:
    """Registration country (or kind of item) from the first digits of an EAN-13 or UPC-A."""
    digits = re.sub(r"\D", "", code)
    if len(digits) == 12:
        digits = "0" + digits
    if len(digits) not in (13, 8) or len(digits) == 8:
        return None
    prefix = int(digits[:3])
    table = _load("gs1")
    for lo, hi, name in table.get("special", []):
        if lo <= prefix <= hi:
            return name
    for lo, hi, name in table.get("ranges", []):
        if lo <= prefix <= hi:
            return name
    return None


def describe_code(fmt: str, text: str) -> tuple[str, str | None, dict | None, list[FollowUp]]:
    """(one-line description, country phrase or None, geo dict or None, follow-ups) with sensitive content withheld."""
    t = text.strip()
    low = t.lower()
    if low.startswith(("http://", "https://")):
        u = urlparse(t)
        host = u.hostname or "unknown"
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        country = None
        try:
            ccs = _load("codes").get("cctlds", {})
            country = ccs.get(tld.lower())
        except Exception:
            pass
        extra = " with a long path or query (withheld)" if (u.path and len(u.path) > 1) or u.query else ""
        return f"a **web address** on **{host}**{extra}", country, None, []
    m = GEO_URI.match(t)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        g = hint(f"QR location ({lat:.2f}, {lon:.2f})", macros_for_latlon(lat, lon), 0.9)
        return (f"a **map location** (**{lat:.5f}, {lon:.5f}**)", None, g,
                [FollowUp("OpenStreetMap at these coordinates", f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}")])
    if low.startswith("wifi:"):
        return "a **Wi-Fi network** login (name and password withheld)", None, None, []
    if low.startswith(("begin:vcard", "mecard:")):
        return "a **contact card** (contents withheld)", None, None, []
    if low.startswith(("tel:", "sms:", "smsto:")):
        return "a **phone number** link (withheld)", None, None, []
    if low.startswith("mailto:") or re.match(r"^[^@\s]+@[^@\s]+\.\w+$", t):
        return "an **e-mail address** (withheld)", None, None, []
    if fmt in ("EAN13", "UPCA", "EAN8", "UPCE") or re.fullmatch(r"\d{12,13}", t):
        c = gs1_country(t)
        return ("a **product barcode**" + (f" whose prefix is registered to **{c}**" if c else ""), c if c and "(" not in c else None, None, [])
    return f"a **{fmt} code** holding {len(t)} characters of text (contents not shown)", None, None, []


class BarcodesProbe(Probe):
    name = "barcodes"
    group = "Text and visuals"
    media = (MediaType.IMAGE,)
    timeout = 90.0

    def run(self, path: Path) -> Finding:
        try:
            import zxingcpp
        except ImportError:
            return Finding(probe=self.name, detected=False, status="not_run",
                           detail="Reading QR codes and barcodes is an add-on: install it from Add-ons in the app, or run: "
                                  "mitraav-setup --install codes")
        try:
            with Image.open(path) as im:
                gray = np.asarray(im.convert("L"))
        except Exception as e:
            return Finding(probe=self.name, detected=False, status="not_run", detail=f"cannot read the picture: {e}")
        found, arr = [], gray
        for scale in (1.0, 0.5, 2.0):
            arr = gray if scale == 1.0 else np.asarray(Image.fromarray(gray).resize(
                (max(8, int(gray.shape[1] * scale)), max(8, int(gray.shape[0] * scale)))))
            if min(arr.shape) < 24 or max(arr.shape) > 9000:
                continue
            found = zxingcpp.read_barcodes(arr)
            if found:
                break
        if not found:
            return Finding(probe=self.name, detected=False, info=True,
                           summary="No QR code or barcode could be read (too small, blurred, curved or at a steep angle).")
        notes, ups, geo, boxes, strength = [], [], None, [], 0.5
        h, w = arr.shape                                  # the scale the code was found at
        for b in found[:6]:
            fmt = str(b.format).split(".")[-1]
            text, country, g, fu = describe_code(fmt, b.text)
            notes.append(text)
            ups += fu
            if g:
                geo, strength = g, 0.9
            elif country and geo is None:
                macros = macros_for_phrase(country)
                geo = hint(f"barcode: {country}", macros, 0.2) if macros else None
            try:
                pts = [b.position.top_left, b.position.top_right, b.position.bottom_left, b.position.bottom_right]
                xs, ys = [p.x for p in pts], [p.y for p in pts]
                boxes.append(dict(label=fmt, x0=max(0.0, min(xs) / w), y0=max(0.0, min(ys) / h),
                                  x1=min(1.0, max(xs) / w), y1=min(1.0, max(ys) / h)))
            except Exception:
                pass
        return Finding(probe=self.name, detected=True, strength=strength, narrows=0.3, effort=1, constrains=["origin", "location"],
                       follow_ups=ups, boxes=[b for b in boxes if b["x1"] > b["x0"] and b["y1"] > b["y0"]],
                       meta={"geo": geo} if geo else {},
                       summary=" • ".join(notes) + " • Read offline; nothing was opened. A code can be printed anywhere, "
                       "so it points to its issuer, not necessarily to where the picture was taken.")
