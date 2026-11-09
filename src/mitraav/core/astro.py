"""Sun position and moon phase from a place and a time (NOAA low-precision formulas, about 0.1 degree for
the sun and a few percent for the moon's lit fraction): enough to ask whether a photo's light and shadows
fit the date, time and place recorded in its metadata. Offline, no ephemeris files."""
from __future__ import annotations

import math
from datetime import UTC, datetime


def julian_day(dt: datetime) -> float:
    dt = dt.astimezone(UTC)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60) / 60) / 24
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def sun_position(lat: float, lon: float, when: datetime) -> tuple[float, float]:
    """(elevation, azimuth) in degrees; azimuth is clockwise from north. `when` must be timezone-aware."""
    jd = julian_day(when)
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    omega = 125.04 - 1934.136 * t
    lam = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(omega)))
    eps = math.radians(23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
                       + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(eps) * math.sin(lam))
    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eqt = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                           - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    utc = when.astimezone(UTC)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    tst = (minutes + eqt + 4 * lon) % 1440
    ha = math.radians(tst / 4 - 180)
    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elev = 90 - math.degrees(zen)
    denom = math.cos(latr) * math.sin(zen)
    if abs(denom) < 1e-9:
        az = 180.0
    else:
        cos_az = (math.sin(decl) - math.sin(latr) * cos_zen) / denom
        az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        if ha > 0:                                   # afternoon: the sun is west of south/north
            az = 360 - az
    return elev, az % 360


def moon_phase(when: datetime) -> tuple[float, bool]:
    """(lit fraction 0..1, waxing?) from the mean synodic month (good to a few percent)."""
    age = (julian_day(when) - 2451550.1) % 29.530588853
    return (1 - math.cos(2 * math.pi * age / 29.530588853)) / 2, age < 14.765


COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")


def compass(az: float) -> str:
    return COMPASS[int(((az % 360) + 22.5) // 45) % 8]
