"""Scan one frame (or a region of it) with the picture probes, for more control than a whole-file scan.

Pick the moment in the player, press "Scan this frame", optionally drag a rectangle around a sign,
animal or other small detail, and the picture probes run on just that. Small selections are enlarged
and sharpened first, which is what lets the text reader see small background signs.
"""
from __future__ import annotations

import html
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
)

from ..core.finding import Finding, MediaType
from ..core.frames import frame_at
from ..core.runner import discover_probes, run_probe
from .spectro_view import fmt_time

ZOOM_TARGET = 1600          # a selection is enlarged until its long side is about this many pixels
MAX_ZOOM = 6.0


def prepare_crop(src: Path, box: tuple[float, float, float, float] | None, out_dir: Path) -> Path:
    """The picture to scan: the whole image, or the selected region (fractions x0, y0, x1, y1)
    enlarged and sharpened so small text is legible to the reader."""
    if box is None:
        return src
    with Image.open(src) as im:
        im = im.convert("RGB")
        x0, y0, x1, y1 = box
        crop = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
    s = max(1.0, min(MAX_ZOOM, ZOOM_TARGET / max(crop.size)))
    if s > 1.2:
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=3, percent=150))
    out = out_dir / f"region-{int(x0 * 1000)}-{int(y0 * 1000)}-{int(x1 * 1000)}-{int(y1 * 1000)}.png"
    crop.save(out)
    return out


def image_probes(file_level: bool = True) -> list[type]:
    """Picture probes. A video frame or a region has no EXIF or JPEG history of its own, so the
    file-level group is left out there."""
    return sorted((c for c in discover_probes() if MediaType.IMAGE in c.media
                   and (file_level or c.group != "File and metadata")), key=lambda c: c.order)


class _Worker(QThread):
    result = Signal(object)

    def __init__(self, path: Path, file_level: bool):
        super().__init__()
        self.path = path
        self.file_level = file_level

    def run(self) -> None:
        for cls in image_probes(self.file_level):
            self.result.emit(run_probe(cls, self.path))


class _Picker(QLabel):
    """The frame, scaled to fit, with a drag-to-select rectangle."""
    changed = Signal()

    def __init__(self, pix: QPixmap):
        super().__init__()
        self.src = pix
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(320, 220)
        self.setCursor(Qt.CrossCursor)
        self.band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.fractions: tuple[float, float, float, float] | None = None

    def _shown(self) -> QRect:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return QRect()
        return QRect((self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2, pm.width(), pm.height())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.setPixmap(self.src.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.clear_selection()

    def clear_selection(self) -> None:
        self.band.hide()
        self.fractions = None
        self.changed.emit()

    def mousePressEvent(self, e) -> None:
        self.origin = e.position().toPoint()
        self.band.setGeometry(QRect(self.origin, QSize()))
        self.band.show()

    def mouseMoveEvent(self, e) -> None:
        self.band.setGeometry(QRect(self.origin, e.position().toPoint()).normalized().intersected(self._shown()))

    def mouseReleaseEvent(self, e) -> None:
        r = QRect(self.origin, e.position().toPoint()).normalized().intersected(self._shown())
        shown = self._shown()
        if r.width() < 12 or r.height() < 12 or shown.isEmpty():
            self.clear_selection()
            return
        self.band.setGeometry(r)
        self.fractions = ((r.left() - shown.left()) / shown.width(), (r.top() - shown.top()) / shown.height(),
                          (r.right() - shown.left()) / shown.width(), (r.bottom() - shown.top()) / shown.height())
        self.changed.emit()


class FrameDialog(QDialog):
    def __init__(self, path: Path, media: MediaType, seconds: float, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.media = media
        self.src_path = path
        self.workdir = Path(tempfile.mkdtemp(prefix="mitraav-frame-"))
        self.frame_path: Path | None = path if media == MediaType.IMAGE else frame_at(path, seconds)
        self.worker: _Worker | None = None
        self.setWindowTitle(f"{path.name}" + (f" at {fmt_time(seconds)}" if media == MediaType.VIDEO else ""))
        lay = QVBoxLayout(self)
        if self.frame_path is None:
            lay.addWidget(QLabel("Could not extract this frame (is ffmpeg installed?)"))
            self.picker = None
            return
        self.picker = _Picker(QPixmap(str(self.frame_path)))
        self.picker.changed.connect(self._sel_changed)
        lay.addWidget(self.picker, 3)
        row = QHBoxLayout()
        self.whole_btn = QPushButton("Scan whole frame")
        self.whole_btn.clicked.connect(lambda: self.scan(None))
        self.region_btn = QPushButton("Scan selection")
        self.region_btn.setEnabled(False)
        self.region_btn.clicked.connect(lambda: self.scan(self.picker.fractions))
        hint = QLabel("Drag a rectangle around a sign or detail to scan just that, enlarged.")
        hint.setObjectName("hint")
        row.addWidget(self.whole_btn)
        row.addWidget(self.region_btn)
        row.addWidget(hint, 1)
        lay.addLayout(row)
        self.out = QTextBrowser()
        self.out.setOpenLinks(False)
        self.out.anchorClicked.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
        self.out.setHtml("<p>Nothing scanned yet.</p>")
        lay.addWidget(self.out, 2)
        self.resize(900, 780)

    def _sel_changed(self) -> None:
        self.region_btn.setEnabled(self.picker.fractions is not None)

    def scan(self, box) -> None:
        if self.worker and self.worker.isRunning():
            return
        target = prepare_crop(self.frame_path, box, self.workdir)
        self.rows: list[str] = []
        self.out.setHtml("<p>Scanning " + ("the selection" if box else "the whole frame") + "...</p>")
        self.whole_btn.setEnabled(False)
        self.region_btn.setEnabled(False)
        self.worker = _Worker(target, self.media == MediaType.IMAGE and box is None)
        self.worker.result.connect(self._found)
        self.worker.finished.connect(self._done)
        self.worker.start()

    def _found(self, f: Finding) -> None:
        from .main_window import DISPLAY, _badge, _rich
        label, color = _badge(f)
        name = html.escape(DISPLAY.get(f.probe, f.probe.replace("_", " ")))
        body = _rich(f.summary) if f.summary else html.escape(f.detail or "")
        if f.status == "ok" and not f.detected and not f.summary:
            body = "nothing found"
        self.rows.append(f'<p><b>{name}</b> <span style="color:{color}">{html.escape(label)}</span><br>{body}</p>')
        self.out.setHtml("".join(self.rows))

    def _done(self) -> None:
        self.whole_btn.setEnabled(True)
        self._sel_changed()
        if not self.rows:
            self.out.setHtml("<p>No picture probes available.</p>")

    def closeEvent(self, e) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(e)
