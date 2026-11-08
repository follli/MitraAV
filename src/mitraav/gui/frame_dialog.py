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
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..core.finding import Finding, MediaType
from ..core.frames import frame_at
from ..core.runner import discover_probes, run_probe
from . import theme
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
    started_probe = Signal(str)
    result = Signal(object)

    def __init__(self, path: Path, file_level: bool):
        super().__init__()
        self.path = path
        self.file_level = file_level
        self.probes = image_probes(file_level)

    def run(self) -> None:
        for cls in self.probes:
            self.started_probe.emit(cls.name)
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
        row.addWidget(hint, 0)
        hint.hide()                                       # the instruction lives in the results pane instead
        lay.addLayout(row)
        from .main_window import (  # late: main_window imports this module
            ROLE,
            PillDelegate,
        )
        self._role = ROLE
        self.status = QLabel("Nothing scanned yet.")
        self.status.setObjectName("hint")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        row.addWidget(self.status, 1)
        lay.addWidget(self.bar)
        self.table = QTreeWidget()
        self.table.setColumnCount(2)
        self.table.setHeaderLabels(["Probe", "Result"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        self.table.header().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.header().setStretchLastSection(True)
        self.table.setColumnWidth(0, 170)
        self.table.setItemDelegateForColumn(1, PillDelegate(self.table))
        self.table.itemSelectionChanged.connect(self._show_detail)
        self.out = QTextBrowser()
        self.out.setOpenLinks(False)
        self.out.anchorClicked.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
        self.out.document().setDefaultStyleSheet(theme.DETAIL_CSS)
        self.out.setHtml("<p class='muted'>Press Scan whole frame, or drag a rectangle and press Scan selection.</p>")
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.table)
        self.split.addWidget(self.out)
        self.split.setSizes([360, 480])
        lay.addWidget(self.split, 2)
        self.found: dict[str, Finding] = {}
        self.resize(900, 780)

    def _sel_changed(self) -> None:
        self.region_btn.setEnabled(self.picker.fractions is not None)

    def scan(self, box) -> None:
        if self.worker and self.worker.isRunning():
            return
        target = prepare_crop(self.frame_path, box, self.workdir)
        self.worker = _Worker(target, self.media == MediaType.IMAGE and box is None)
        self.found.clear()
        self.table.clear()
        self.items: dict[str, QTreeWidgetItem] = {}
        from .main_window import DISPLAY, C
        for cls in self.worker.probes:
            it = QTreeWidgetItem([DISPLAY.get(cls.name, cls.name.replace("_", " ")), "queued"])
            it.setData(0, Qt.UserRole + 1, cls.name)
            it.setData(1, self._role, C["muted"])
            self.table.addTopLevelItem(it)
            self.items[cls.name] = it
        self.out.setHtml("<p class='muted'>Select a probe to read its result.</p>")
        self.bar.setRange(0, len(self.worker.probes))
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.status.setText(f"Scanning {'the selection' if box else 'the whole frame'}...")
        self.whole_btn.setEnabled(False)
        self.region_btn.setEnabled(False)
        self.worker.started_probe.connect(self._started)
        self.worker.result.connect(self._found)
        self.worker.finished.connect(self._done)
        self.worker.start()

    def _started(self, name: str) -> None:
        from .main_window import C
        it = self.items.get(name)
        if it:
            it.setText(1, "running...")
            it.setData(1, self._role, C["text"])
        self.status.setText(f"Running {it.text(0) if it else name} ({self.bar.value() + 1} of {self.bar.maximum()})...")

    def _found(self, f: Finding) -> None:
        from .main_window import _badge
        label, color = _badge(f)
        it = self.items.get(f.probe)
        if it:
            it.setText(1, label)
            it.setData(1, self._role, color)
        self.found[f.probe] = f
        self.bar.setValue(self.bar.value() + 1)
        sel = self.table.selectedItems()
        if sel and sel[0] is it:
            self._show_detail()

    def _show_detail(self) -> None:
        from .main_window import DISPLAY, _badge, _rich
        sel = self.table.selectedItems()
        if not sel:
            return
        name = sel[0].data(0, Qt.UserRole + 1)
        f = self.found.get(name)
        if f is None:
            self.out.setHtml("<p class='muted'>This probe has not finished yet.</p>")
            return
        label, color = _badge(f)
        parts = [f"<h3>{html.escape(DISPLAY.get(name, name.replace('_', ' ')))} "
                 f'<span style="color:{color}">{html.escape(label)}</span></h3>']
        parts.append(f"<p>{_rich(f.summary) if f.summary else html.escape(f.detail or 'Nothing found.')}</p>")
        if f.follow_ups:
            parts.append("<p><b>Follow up with:</b></p><ul>" + "".join(
                f'<li><a href="{html.escape(u.url, quote=True)}">{html.escape(u.name)}</a></li>' if u.url
                else f"<li>{html.escape(u.name)}</li>" for u in f.follow_ups) + "</ul>")
        self.out.setHtml("".join(parts))

    def _done(self) -> None:
        self.whole_btn.setEnabled(True)
        self._sel_changed()
        self.bar.setVisible(False)
        leads = sum(1 for f in self.found.values() if f.detected and f.status == "ok" and not f.quiet
                    and not f.inconclusive)
        self.status.setText(f"Done: {leads} lead(s) from {len(self.found)} probe(s). Select a row for details.")
        first = next((self.table.topLevelItem(i) for i in range(self.table.topLevelItemCount())
                      if (f := self.found.get(self.table.topLevelItem(i).data(0, Qt.UserRole + 1))) and f.detected), None)
        if first is not None:
            first.setSelected(True)                      # open the best-looking result straight away

    def closeEvent(self, e) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(e)
