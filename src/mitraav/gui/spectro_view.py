from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.audio import MAX_SECONDS, load_audio, sound_start, total_seconds
from ..core.spectro import compute, to_rgb
from . import theme

C = theme.COLORS
FREQ_CHOICES = [("0-500 Hz", 500.0), ("0-2 kHz", 2000.0), ("0-8 kHz", 8000.0), ("Full range", 0.0)]
SPAN_CHOICES = [("Signal window", 0.0), ("Wider (about 2 minutes)", 120.0), ("Widest (4 minutes)", 240.0)]
MAX_VIEW = 240.0   # longest stretch shown at once (s)


def nice_ticks(a: float, b: float, target: int = 6) -> list[float]:
    """Round tick positions covering [a, b]."""
    span = max(b - a, 1e-9)
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = np.ceil(a / step) * step
    return [float(v) for v in np.arange(first, b + step * 0.001, step)]


def fmt_time(t: float, show_fraction: bool = False) -> str:
    """File time as m:ss (h:mm:ss past an hour); plain seconds under a minute."""
    if t < 60:
        return f"{t:.1f} s" if show_fraction else f"{t:.0f} s"
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    sec = f"{s:04.1f}" if show_fraction else f"{s:02.0f}"
    return f"{int(h)}:{int(m):02d}:{sec}" if h else f"{int(m)}:{sec}"


def fmt_freq(f: float, unit: bool = True) -> str:
    if f >= 1000:
        v = f / 1000
        return f"{v:.1f} kHz" if unit else f"{v:g}"
    return f"{f:.0f} Hz" if unit else f"{f:.0f}"


class SpectroCanvas(QWidget):
    """The spectrogram image with labelled axes, gridlines and a crosshair."""
    hover = Signal(float, float, bool)          # time, frequency, inside the plot

    LEFT, RIGHT, TOP, BOTTOM = 62, 14, 10, 34

    def __init__(self):
        super().__init__()
        self.pix: QPixmap | None = None
        self.t0, self.t1, self.f0, self.f1 = 0.0, 1.0, 0.0, 1.0
        self.cursor: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, pix: QPixmap, t0: float, t1: float, f0: float, f1: float) -> None:
        self.pix, self.t0, self.t1, self.f0, self.f1 = pix, t0, t1, f0, f1
        self.update()

    def plot_rect(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(10, self.width() - self.LEFT - self.RIGHT),
                      max(10, self.height() - self.TOP - self.BOTTOM))

    def to_tf(self, p: QPointF) -> tuple[float, float]:
        r = self.plot_rect()
        fx = (p.x() - r.left()) / r.width()
        fy = (p.y() - r.top()) / r.height()
        return self.t0 + fx * (self.t1 - self.t0), self.f0 + (1 - fy) * (self.f1 - self.f0)

    def mouseMoveEvent(self, e) -> None:
        self.cursor = e.position()
        inside = self.plot_rect().contains(self.cursor)
        t, f = self.to_tf(self.cursor)
        self.hover.emit(t, f, inside)
        self.update()

    def leaveEvent(self, e) -> None:
        self.cursor = None
        self.hover.emit(0.0, 0.0, False)
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(C["bg"]))
        r = self.plot_rect()
        if self.pix is None:
            p.setPen(QColor(C["muted"]))
            p.drawText(r, Qt.AlignCenter, "No image")
            return
        p.drawPixmap(r.toRect(), self.pix)
        small = QFont(self.font())
        small.setPointSizeF(max(8.0, self.font().pointSizeF() - 1))
        p.setFont(small)
        grid = QColor(255, 255, 255, 38)
        axis = QColor(C["muted"])
        long_view = (self.t1 - self.t0) >= 60 or self.t0 >= 60
        for f in nice_ticks(self.f0, self.f1, 6):
            y = r.bottom() - (f - self.f0) / (self.f1 - self.f0) * r.height()
            p.setPen(QPen(grid, 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(axis)
            p.drawText(QRectF(0, y - 9, self.LEFT - 6, 18), Qt.AlignRight | Qt.AlignVCenter, fmt_freq(f))
        for t in nice_ticks(self.t0, self.t1, 7):
            x = r.left() + (t - self.t0) / (self.t1 - self.t0) * r.width()
            p.setPen(QPen(grid, 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(axis)
            p.drawText(QRectF(x - 40, r.bottom() + 4, 80, 18), Qt.AlignHCenter | Qt.AlignTop,
                       fmt_time(t) if long_view else f"{t:.0f} s")
        p.setPen(QPen(QColor(C["line"]), 1))
        p.drawRect(r)
        if self.cursor is not None and r.contains(self.cursor):          # crosshair
            p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine))
            p.drawLine(QPointF(self.cursor.x(), r.top()), QPointF(self.cursor.x(), r.bottom()))
            p.drawLine(QPointF(r.left(), self.cursor.y()), QPointF(r.right(), self.cursor.y()))


class SpectrogramDialog(QDialog):
    """Shows the identified signal when probes supplied regions, else the first stretch of sound
    (leading digital silence is skipped). Times are file times."""

    def __init__(self, path: Path, parent=None, regions: list[dict] | None = None):
        super().__init__(parent)
        self.path = Path(path)
        self.regions = regions or []
        self.setWindowTitle(f"Signal - {self.path.name}")
        self.resize(880, 560)
        if not self.regions:
            load_audio(self.path, duration=1.0)          # raise AudioError early if unreadable
        self.total = total_seconds(self.path)

        self.region_box = QComboBox()
        for r in self.regions:
            self.region_box.addItem(r["label"], r)
        self.region_box.addItem("Start of the sound (no identified signal)", None)
        self.region_box.currentIndexChanged.connect(self._region_changed)
        self.freq_box = QComboBox()
        self.span_box = QComboBox()
        for label, val in SPAN_CHOICES:
            self.span_box.addItem(label, val)
        self.flatten = QCheckBox("Flatten background")
        self.flatten.setChecked(True)

        self.info = QLabel("")
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)
        self.readout = QLabel("Move the pointer over the image")
        big = QFont(self.font())
        big.setPointSizeF(big.pointSizeF() + 3)
        big.setBold(True)
        self.readout.setFont(big)
        self.canvas = SpectroCanvas()
        self.canvas.hover.connect(self._hover)

        top = QHBoxLayout()
        for label, w, stretch in (("Signal", self.region_box, 1), ("Frequency", self.freq_box, 0),
                                  ("Time span", self.span_box, 0)):
            lab = QLabel(label)
            lab.setObjectName("hint")
            top.addWidget(lab)
            top.addWidget(w, stretch)
        top.addWidget(self.flatten)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.info)
        lay.addWidget(self.readout)
        lay.addWidget(self.canvas, 1)

        self._fill_freq()
        self.freq_box.currentIndexChanged.connect(self.render)
        self.span_box.currentIndexChanged.connect(self.render)
        self.flatten.toggled.connect(self.render)
        self.render()

    # --- controls ---
    def _fill_freq(self) -> None:
        self.freq_box.blockSignals(True)
        self.freq_box.clear()
        if self.region_box.currentData():
            self.freq_box.addItem("Signal band", None)
        for label, val in FREQ_CHOICES:
            self.freq_box.addItem(label, val)
        if not self.region_box.currentData():
            self.freq_box.setCurrentIndex(2)             # 0-8 kHz
        self.freq_box.blockSignals(False)
        self.span_box.setEnabled(bool(self.region_box.currentData()))

    def _region_changed(self) -> None:
        self._fill_freq()
        self.render()

    def _hover(self, t: float, f: float, inside: bool) -> None:
        self.readout.setText(f"{fmt_time(t, True)}   ·   {fmt_freq(f)}" if inside
                             else "Move the pointer over the image")

    # --- drawing ---
    def _window(self) -> tuple[float, float, float | None, float | None]:
        """(t0, duration, fmin, fmax) for the current choices; None frequency = signal band."""
        region = self.region_box.currentData()
        fsel = self.freq_box.currentData()
        if region:
            t0, t1 = float(region["t0"]), float(region["t1"])
            want = float(self.span_box.currentData() or 0.0)
            if want > (t1 - t0):                         # widen around the signal's centre
                mid = (t0 + t1) / 2
                t0, t1 = max(0.0, mid - want / 2), min(self.total, mid + want / 2)
            dur = min(t1 - t0, MAX_VIEW)
            if fsel is None:
                return t0, dur, float(region["f0"]), float(region["f1"])
            return t0, dur, 0.0, fsel or None
        t0 = sound_start(self.path)
        return t0, min(MAX_SECONDS, MAX_VIEW), 0.0, fsel or None

    def render(self) -> None:
        t0, dur, fmin, fmax = self._window()
        x, sr = load_audio(self.path, offset=t0, duration=dur)
        region = self.region_box.currentData()
        hi_f = fmax if fmax else sr / 2
        # tonal low-frequency signals want long windows; chirps and calls want short ones
        win = 0.6 if hi_f < 1500 else 0.2 if hi_f < 4000 else 0.06
        nper = int(min(65536, max(256, 2 ** round(np.log2(sr * win)))))
        if not region:
            nper = 4096
        f, t, db = compute(x, sr, nper)
        hi = fmax if fmax else float(f[-1])
        rows = (f >= (fmin or 0.0)) & (f <= hi)
        db = db[rows]
        if self.flatten.isChecked():
            db = db - np.median(db, axis=1, keepdims=True)          # drop stationary lines and hum
        rgb = to_rgb(db, 30.0 if region else 20.0, 99.7)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        fr = f[rows]
        t1 = t0 + len(x) / sr
        self.canvas.set_image(QPixmap.fromImage(img), t0, t1, float(fr[0]), float(fr[-1]))
        name = self.region_box.currentText()
        self.info.setText(f"{name}.   Time {fmt_time(t0, True)} to {fmt_time(t1, True)}   ·   "
                          f"frequency {fmt_freq(float(fr[0]))} to {fmt_freq(float(fr[-1]))}   ·   "
                          f"{sr} Hz audio")
