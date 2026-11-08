from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSplitterHandle,
    QStyle,
    QStyledItemDelegate,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import history
from ..core.audio import AudioError
from ..core.context import from_findings as context_from
from ..core.finding import Finding, MediaType, plain
from ..core.runner import media_of, planned, scan
from . import theme
from .addons_dialog import AddonsDialog
from .frame_dialog import FrameDialog
from .history_dialog import HistoryDialog
from .preview import MediaPreview, _FullImage
from .spectro_view import SpectrogramDialog, fmt_time

C = theme.COLORS
COLUMNS = ["Probe", "Result", ""]
FOCUS_COL = 2
DISPLAY = {"tones_chimes": "tones and chimes", "sound_events": "sound overview"}      # names shown in place of the probe id
ROLE = Qt.UserRole
NAME_ROLE = Qt.UserRole + 1
LAST_GROUP = "File and metadata"

COLUMN_TIPS = [
    "The probe that ran.",
    "lead = the probe found something. info = it ran and found nothing notable (click for details; "
    "this is not proof of absence).",
    "Row action: Spectro opens the spectrogram on the signal, Image shows the picture full size, "
    "Map opens the coordinates in OpenStreetMap.",
]
EXPLAIN = ("A <b>lead</b> means the probe found something worth a look; <b>info</b> means it ran and found "
           "nothing notable. Neither is proof: a clear signal can still say little about where it was recorded.")


class ScanWorker(QThread):
    started_probe = Signal(str)
    found = Signal(object)

    def __init__(self, path: Path, only: set[str] | None = None, context: dict | None = None):
        super().__init__()
        self.path = path
        self.only = only
        self.context = context

    def run(self) -> None:
        scan(self.path, on_result=self.found.emit, on_start=self.started_probe.emit,
             only=self.only, context=self.context)


def _badge(f: Finding) -> tuple[str, str]:
    """Pill text and colour. Colours are only used for results. Blue means 'information': the probe
    ran and has something to read (including 'nothing notable'), so it is never mistaken for a lead
    or a failure, and every row is clickable."""
    if f.status != "ok":
        return f.status.replace("_", " "), C["amber"] if f.status == "not_run" else C["red"]
    if f.inconclusive:
        return "inconclusive", C["amber"]
    if f.detected and f.quiet:
        return f.pill or "context", C["muted"]
    if f.detected:
        return ("lead · very low", C["amber"]) if f.very_low else ("lead", C["green"])
    return f.pill or "info", C["blue"]


def _rich(text: str) -> str:
    """Escape, then turn **key words** into bold."""
    parts = html.escape(text).split("**")
    out = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    muted = out.split("~~")                           # ~~text~~ = weaker items, shown muted but kept
    return "".join(f'<span class="muted">{p}</span>' if i % 2 else p for i, p in enumerate(muted))


def _is_active(f: Finding) -> bool:
    """False for probes that ran but have nothing to act on (info, not run, inconclusive, error)."""
    return f.status == "ok" and not f.inconclusive and (f.detected or f.quiet)


def row_action(f: Finding, media: MediaType | None) -> str:
    """The button a result row offers: what is most useful to open for this finding (or "")."""
    if f.status != "ok" or f.inconclusive:
        return ""
    if f.regions and media in (MediaType.AUDIO, MediaType.VIDEO):
        return "Spectro"
    if any(u.url.startswith("https://www.openstreetmap.org/?mlat") for u in f.follow_ups):
        return "Map"
    if media == MediaType.IMAGE and (f.boxes or (f.group in ("Text and visuals", "Visual context")
                                                 and (f.detected or f.quiet))):
        return "Image"
    return ""


def map_url(f: Finding) -> str:
    return next((u.url for u in f.follow_ups if u.url.startswith("https://www.openstreetmap.org/?mlat")), "")


def merge_play_links(regions: list[dict], gap: float = 10.0) -> list[tuple[float, str]]:
    """One play link per cluster of events that start within `gap` seconds of each other."""
    rows = sorted((_play_at(r), r["label"]) for r in regions)
    out: list[list] = []
    for t, label in rows:
        if out and t - out[-1][0] < gap:
            out[-1][2] += 1
        else:
            out.append([t, label, 1])
    return [(t, label if n == 1 else f"{label} and {n - 1} more nearby") for t, label, n in out]


def _play_at(region: dict) -> float:
    """Where playback should start for a region: just before the event when a probe says so."""
    return max(0.0, float(region.get("t_play", region["t0"])))


def _is_lead(f: Finding) -> bool:
    return f.detected and f.status == "ok" and not f.quiet and not f.inconclusive


def _base(d: QStyledItemDelegate, p, opt, idx) -> None:
    """Draw only the row background/selection, no text."""
    o = type(opt)(opt)
    d.initStyleOption(o, idx)
    o.text = ""
    (o.widget.style() if o.widget else QApplication.style()).drawControl(
        QStyle.CE_ItemViewItem, o, p, o.widget)


class PillDelegate(QStyledItemDelegate):
    """Status shown as a tinted pill; the colour travels in Qt.UserRole."""

    def paint(self, p, opt, idx) -> None:
        _base(self, p, opt, idx)
        text = idx.data()
        if not text:
            return
        col = QColor(idx.data(ROLE) or C["muted"])
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(opt.rect).adjusted(8, 8, -8, -8)
        r.setWidth(min(r.width(), p.fontMetrics().horizontalAdvance(text) + 22))
        bg = QColor(col)
        bg.setAlpha(40)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(col)
        p.drawText(r, Qt.AlignCenter, text)
        p.restore()


class FocusDelegate(QStyledItemDelegate):
    """A small action pill (Spectro, Image, Map) on rows that have something to open."""

    def paint(self, p, opt, idx) -> None:
        _base(self, p, opt, idx)
        if not idx.data(ROLE):
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(opt.rect).adjusted(6, 9, -14, -9)
        p.setPen(QColor(C["muted"]))
        p.setBrush(QColor(C["muted"]))                  # filled with the outline colour: a button
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        font = QFont(p.font())
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(C["bg"]))                       # dark text on the grey fill
        p.drawText(r, Qt.AlignCenter, str(idx.data(ROLE)))
        p.restore()


class _Grip(QSplitterHandle):
    """A splitter handle with a small bar in the middle, so it is obvious it can be dragged."""

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        col = QColor(C["muted"])
        col.setAlpha(255 if self.underMouse() else 150)
        p.setBrush(col)
        p.setPen(Qt.NoPen)
        r = QRectF(self.rect())
        long_, short = 44.0, 4.0
        if self.orientation() == Qt.Horizontal:        # a vertical divider: a tall thin bar
            p.drawRoundedRect(QRectF(r.center().x() - short / 2, r.center().y() - long_ / 2, short, long_), 2, 2)
        else:
            p.drawRoundedRect(QRectF(r.center().x() - long_ / 2, r.center().y() - short / 2, long_, short), 2, 2)


class GripSplitter(QSplitter):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.setHandleWidth(12)

    def createHandle(self):
        return _Grip(self.orientation(), self)


class ProbeTree(QTreeWidget):
    """The results tree; the pointer becomes a hand over an action pill."""

    def mouseMoveEvent(self, e) -> None:
        idx = self.indexAt(e.position().toPoint())
        hand = idx.isValid() and idx.column() == FOCUS_COL and bool(idx.data(ROLE))
        self.viewport().setCursor(Qt.PointingHandCursor if hand else Qt.ArrowCursor)
        super().mouseMoveEvent(e)


class ElidedLabel(QLabel):
    """A label that shortens its text in the middle to fit, keeping the full text available."""

    def __init__(self, text: str = ""):
        super().__init__()
        self._full = ""
        self.setFullText(text)

    def setFullText(self, text: str) -> None:
        self._full = text
        self._elide()

    def fullText(self) -> str:
        return self._full

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        room = max(20, self.width() - 40)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, room))

    def minimumSizeHint(self):
        return QSize(80, super().minimumSizeHint().height())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MitraAV")
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self.findings: list[Finding] = []
        self.by_name: dict[str, Finding] = {}
        self.quiet_groups: set[str] = set()
        self.plan: list[tuple[str, str]] = []        # (probe name, group) the scan will run
        self.running: str | None = None
        self.collapsed: set[str] = {"Visual context"}     # quiet context starts folded away
        self.worker: ScanWorker | None = None
        self.scanning = False
        self.active: set[str] | None = None            # probes of a partial run (None = all)
        self.path: Path | None = None
        self._tick = 0

        title = QLabel("MitraAV")
        title.setObjectName("title")
        sub = QLabel("OSINT lead finder for audio, image and video")
        sub.setObjectName("subtitle")
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(title)
        names.addWidget(sub)
        open_btn = QPushButton("Open file")
        open_btn.setObjectName("segLeft")
        open_btn.clicked.connect(self.choose_file)
        addons_btn = QPushButton("Add-ons")
        addons_btn.setObjectName("secondary")
        addons_btn.clicked.connect(self.open_addons)
        hist_btn = QPushButton("History")
        hist_btn.setObjectName("segRight")
        hist_btn.clicked.connect(self.open_history)
        split = QHBoxLayout()                           # Open file | History, one split pill
        split.setSpacing(0)
        split.addWidget(open_btn)
        split.addWidget(hist_btn)
        self.drop = ElidedLabel("Drop a file here")       # the drop target: also shows the file name
        self.drop.setObjectName("dropzone")
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setFixedHeight(38)
        self.drop.setMaximumWidth(420)
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addLayout(names)
        header.addWidget(self.drop, 1)
        self.start_btn = QPushButton("Start scan")
        self.start_btn.setEnabled(False)                # needs a file first
        self.start_btn.clicked.connect(self.run_scan)
        header.addWidget(self.start_btn)
        header.addLayout(split)
        header.addWidget(addons_btn)

        self.preview = MediaPreview()
        self.preview.frame_requested.connect(self.open_frame)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)

        self.focus_ok = False                            # audio or video: a signal can be shown
        self.media: MediaType | None = None
        self.tree = ProbeTree()
        self.tree.setMouseTracking(True)
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        for i, tip in enumerate(COLUMN_TIPS):
            self.tree.headerItem().setToolTip(i, tip)
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        hh = self.tree.header()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(FOCUS_COL, QHeaderView.Fixed)
        self.tree.setColumnWidth(0, 200)
        self.tree.setColumnWidth(FOCUS_COL, 74)
        self.tree.setItemDelegateForColumn(FOCUS_COL, FocusDelegate(self.tree))
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemDoubleClicked.connect(self._item_double_clicked)
        self.tree.setItemDelegateForColumn(1, PillDelegate(self.tree))
        self.tree.itemSelectionChanged.connect(self.show_detail)
        self.tree.itemCollapsed.connect(lambda it: self.collapsed.add(it.text(0)))
        self.tree.itemExpanded.connect(lambda it: self.collapsed.discard(it.text(0)))

        self.detail = QTextBrowser()
        self.detail.setOpenLinks(False)                 # we route links: play-from-here or the browser
        self.detail.anchorClicked.connect(self._link_clicked)
        self.detail.document().setDefaultStyleSheet(theme.DETAIL_CSS)
        self.detail.setHtml(self._placeholder())

        right = GripSplitter(Qt.Vertical)                   # preview above the details; drag to resize
        right.setChildrenCollapsible(False)
        right.addWidget(self.preview)
        right.addWidget(self.detail)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)
        right.setSizes([260, 440])

        split = GripSplitter(Qt.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        split.setSizes([640, 440])

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(20, 16, 20, 10)
        lay.setSpacing(14)
        lay.addLayout(header)
        lay.addWidget(self.progress)
        lay.addWidget(split, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

        self.timer = QTimer(self)                    # animates the "running" dots
        self.timer.setInterval(450)
        self.timer.timeout.connect(self._animate)

    @staticmethod
    def _placeholder() -> str:
        return ('<p class="muted">Select a probe on the left to see what it found and '
                f'where to look next.</p><p class="muted">{EXPLAIN}</p>')

    # --- input ---
    def _drop_active(self, on: bool) -> None:
        self.drop.setProperty("active", on)
        self.drop.style().unpolish(self.drop)
        self.drop.style().polish(self.drop)

    def choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open file to scan")
        if name:
            self.load_file(Path(name))

    def open_addons(self) -> None:
        AddonsDialog(self).exec()

    def open_history(self) -> None:
        dlg = HistoryDialog(self)
        if dlg.exec() and dlg.selected_path:
            self.load_file(dlg.selected_path)

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            self._drop_active(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e) -> None:
        self._drop_active(False)

    def dropEvent(self, e) -> None:
        self._drop_active(False)
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.load_file(Path(urls[0]))

    # --- scanning ---
    def start_scan(self, path: Path) -> None:
        """Load a file and scan it at once (command line use)."""
        if self.load_file(path):
            self.run_scan()

    def load_file(self, path: Path) -> bool:
        """Show a file (preview, the probes that will run) without scanning it: press Start scan."""
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Scan already running, please wait")
            return False
        if not path.is_file():
            self.statusBar().showMessage(f"Not a file: {path}")
            return False
        self.findings.clear()
        self.by_name.clear()
        self.running = None
        self.scanning = False
        self.active = None
        plan = planned(path)
        self.plan = [(cls.name, cls.group) for cls, _ in plan]
        self.quiet_groups = {g for g in {c.group for c, _ in plan}
                             if all(c.quiet for c, _ in plan if c.group == g)}
        self.path = path
        self.drop.setFullText(path.name)
        self.drop.setProperty("loaded", True)
        self._drop_active(False)
        self.drop.setToolTip(str(path))
        self.preview.load(path, media_of(path))
        self.media = media_of(path)
        self.focus_ok = self.media in (MediaType.AUDIO, MediaType.VIDEO)
        self.progress.setVisible(False)
        self.start_btn.setText("Start scan")
        self.start_btn.setEnabled(bool(self.plan))
        self.detail.setHtml(self._placeholder())
        self.refresh()
        self.statusBar().showMessage(
            f"{path.name} loaded: press Start scan" if self.plan else "No probes apply to this file type yet")
        return True

    def run_scan(self, only: set[str] | None = None) -> None:
        """Run every probe, or just those named in `only` (a group's Run button)."""
        path = self.path
        if not path or not self.plan or (self.worker and self.worker.isRunning()):
            return
        if only is not None and not isinstance(only, (set, frozenset)):
            only = None                                  # the button's clicked(bool) lands here
        context = context_from(self.findings) if only is not None else None   # what earlier probes found
        if only is None:
            self.findings.clear()
            self.by_name.clear()
        else:
            self.findings = [f for f in self.findings if f.probe not in only]
            self.by_name = {f.probe: f for f in self.findings}
        self.active = set(only) if only is not None else None
        self.running = None
        self.scanning = True
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Scanning...")
        self.progress.setRange(0, len(only) if only is not None else len(self.plan))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.statusBar().showMessage("Scanning...")
        self.detail.setHtml(self._placeholder())
        self.refresh()
        self.timer.start()
        self.worker = ScanWorker(path, only, context)
        self.worker.started_probe.connect(self.on_started)
        self.worker.found.connect(self.add_finding)
        self.worker.finished.connect(self.scan_done)
        self.worker.start()

    def on_started(self, name: str) -> None:
        self.running = name
        self.statusBar().showMessage(f"Running {DISPLAY.get(name, name.replace('_', ' '))} "
                                     f"({self.progress.value() + 1} of {self.progress.maximum()})...")
        self.refresh()

    def add_finding(self, f: Finding) -> None:
        self.findings = [o for o in self.findings if o.probe != f.probe] + [f]
        self.by_name[f.probe] = f
        if self.running == f.probe:
            self.running = None
        if self.plan:
            self.progress.setValue(min(self.progress.value() + 1, self.progress.maximum()))
        self.refresh()

    def scan_done(self) -> None:
        self.timer.stop()
        self.running = None
        self.scanning = False
        self.start_btn.setText("Rescan")
        self.start_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.refresh()
        if not self.findings:
            self.statusBar().showMessage("No probes apply to this file type yet")
            return
        leads = sum(1 for f in self.findings if _is_lead(f))
        need = sum(1 for f in self.findings if f.status == "not_run" and "add-on" in f.detail.lower())
        self.statusBar().showMessage(
            f"Done: {leads} lead(s) from {len(self.findings)} probe(s)"
            + (f" · {need} probe(s) need add-ons (see Add-ons)" if need else ""))
        if self.path:
            history.add(self.path, leads, len(self.findings))

    def _animate(self) -> None:
        self._tick += 1
        if self.running:
            self.refresh()

    def _item_clicked(self, item, column: int) -> None:
        action = item.data(FOCUS_COL, ROLE) if column == FOCUS_COL else None
        if action == "Run" and item.parent() is None:
            group = item.data(0, NAME_ROLE + 1)
            self.run_scan({n for n, g in self.plan if g == group})
            return
        f = self.by_name.get(item.data(0, NAME_ROLE)) if action else None
        if f is None:
            return
        if action == "Spectro":
            self.open_spectrogram(f.regions)
        elif action == "Map":
            QDesktopServices.openUrl(QUrl(map_url(f)))
        elif action == "Image" and self.path:
            dlg = _FullImage(self.path, self, f.boxes)
            dlg.setAttribute(Qt.WA_DeleteOnClose)
            dlg.setWindowFlag(Qt.Window, True)
            self._keep(dlg)
            dlg.show()

    def _item_double_clicked(self, item, column: int) -> None:
        if column != FOCUS_COL and item.data(FOCUS_COL, ROLE) in ("Spectro", "Image"):   # anywhere on the row
            self._item_clicked(item, FOCUS_COL)

    def open_spectrogram(self, regions: list[dict] | None = None) -> None:
        if not self.path:
            return
        try:
            dlg = SpectrogramDialog(self.path, self, self._regions() if regions is None else regions)
        except AudioError as e:
            self.statusBar().showMessage(str(e))
            return
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowFlag(Qt.Window, True)                 # its own window: several can stay open
        self._keep(dlg)
        dlg.show()

    def open_frame(self, seconds: float) -> None:
        if not self.path or self.media not in (MediaType.IMAGE, MediaType.VIDEO):
            return
        dlg = FrameDialog(self.path, self.media, seconds, self)
        self._keep(dlg)
        dlg.show()

    def _keep(self, dlg) -> None:
        """Hold a reference to an open window; closed windows delete themselves, so drop them on destroy."""
        if not hasattr(self, "_detail_windows"):
            self._detail_windows = []
        self._detail_windows.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._detail_windows.remove(d) if d in self._detail_windows else None)

    def _link_clicked(self, url) -> None:
        if url.scheme() == "mitraav-play":
            try:
                self.preview.play_from(float(url.path()))
            except ValueError:
                pass
            return
        QDesktopServices.openUrl(url)

    def _regions(self) -> list[dict]:
        """The selected lead's signal regions, else those of the best-scoring lead."""
        sel = self._selected()
        if sel and sel.regions:
            return sel.regions
        for f in sorted(self.findings, key=lambda f: -f.score):
            if f.regions and f.detected:
                return f.regions
        return []

    # --- display ---
    def _selected(self) -> Finding | None:
        items = self.tree.selectedItems()
        return self.by_name.get(items[0].data(0, NAME_ROLE)) if items else None

    def _groups(self) -> list[tuple[str, list[tuple[str, Finding | None]]]]:
        """[(group, [(probe name, finding or None while pending)])], best leads first."""
        rows: dict[str, list[tuple[str, Finding | None]]] = {}
        for f in self.findings:
            rows.setdefault(f.group or "Other", []).append((f.probe, f))
        for name, group in self.plan:
            if name not in self.by_name:
                rows.setdefault(group, []).append((name, None))
        for g, items in rows.items():                      # leads, then waiting, then inactive (greyed)
            items.sort(key=lambda r: (2 if r[1] is not None and not _is_active(r[1]) else
                                      1 if r[1] is None else 0,
                                      -(r[1].score if r[1] else 0), r[1].status != "ok" if r[1] else False, r[0]))
        scanning = any(f is None for items in rows.values() for _, f in items)
        order = list(dict.fromkeys(group for _, group in self.plan))

        def key(g: str):
            quiet = g in self.quiet_groups or (all(f is not None and f.quiet for _, f in rows[g]) and bool(rows[g]))
            if scanning:                                   # stable while probes report in: no groups jumping about
                return (g == LAST_GROUP, quiet, order.index(g) if g in order else 99, g)
            best = max((f.score for _, f in rows[g] if f and not f.quiet), default=0.0)
            return (g == LAST_GROUP, quiet, -best, g)
        return [(g, rows[g]) for g in sorted(rows, key=key)]

    def refresh(self) -> None:
        keep = self._selected().probe if self._selected() else None
        self.tree.blockSignals(True)
        self.tree.clear()
        dots = "." * (self._tick % 3 + 1)
        sel_item = None
        for group, items in self._groups():
            leads = sum(1 for _, f in items if f and _is_lead(f))
            ctx = bool(items) and all(f is not None and f.quiet for _, f in items)
            g = QTreeWidgetItem([group, "context only" if ctx else
                                 (f"{leads} of {len(items)} lead{'s' if len(items) != 1 else ''}"
                                  if all(f for _, f in items) else f"{len(items)} probe{'s' if len(items) != 1 else ''}"), ""])
            font = QFont(self.font())
            font.setBold(True)
            g.setFont(0, font)
            g.setFlags(Qt.ItemIsEnabled)
            for c in range(len(COLUMNS)):
                g.setBackground(c, QColor(C["panel2"]))
            g.setForeground(1, QColor(C["muted"]))
            if self.path and not self.scanning and self.plan:
                g.setData(FOCUS_COL, ROLE, "Run")
                g.setToolTip(FOCUS_COL, f"Run only the {group} probes")
            g.setData(0, NAME_ROLE + 1, group)
            self.tree.addTopLevelItem(g)
            for name, f in items:
                it = QTreeWidgetItem([DISPLAY.get(name, name.replace("_", " ")), "", ""])
                it.setData(0, NAME_ROLE, name)
                if f is None:
                    running = name == self.running
                    queued = self.scanning and (self.active is None or name in self.active)
                    it.setText(1, f"running{dots}" if running else ("queued" if queued else "ready"))
                    it.setData(1, ROLE, C["text"] if running else C["muted"])
                    it.setForeground(0, QColor(C["text"] if running else C["muted"]))
                else:
                    label, color = _badge(f)
                    it.setText(1, label)
                    it.setData(1, ROLE, color)
                    it.setToolTip(0, plain(f.summary) or f.detail)
                    kind = self.media if self.media in (MediaType.IMAGE,) else (
                        (self.media or MediaType.AUDIO) if self.focus_ok else self.media)
                    action = row_action(f, kind)
                    if action:
                        it.setData(FOCUS_COL, ROLE, action)
                        it.setToolTip(FOCUS_COL, {"Spectro": "Open the spectrogram on this finding's signal",
                                                  "Image": "Show the picture full size (text regions outlined when found)",
                                                  "Map": "Open these coordinates in OpenStreetMap"}[action])
                    if not _is_active(f):                  # nothing to act on: name greyed, sits at the bottom
                        it.setForeground(0, QColor(C["muted"]))
                    for c in range(len(COLUMNS)):          # finished rows sit slightly lighter than queued ones
                        it.setBackground(c, QColor(C["done"]))
                g.addChild(it)
                if name == keep:
                    sel_item = it
            g.setExpanded(group not in self.collapsed)
        if sel_item:
            sel_item.setSelected(True)
        self.tree.blockSignals(False)

    def show_detail(self) -> None:
        f = self._selected()
        if f is None:
            items = self.tree.selectedItems()
            if items and items[0].parent() is not None:
                self.detail.setHtml("<p class='muted'>This probe has not finished yet.</p>")
            return
        label, color = _badge(f)
        parts = [f"<h3>{html.escape(DISPLAY.get(f.probe, f.probe))} "
                 f'<span style="color:{color}">{html.escape(label)}</span></h3>']
        if f.detected and f.status == "ok" and f.quiet and not f.inconclusive:
            parts.append("<p class='muted'>Context only: this reports what the picture shows to the "
                         "eye, so it is not counted as a lead.</p>")
        elif f.detected and f.status == "ok" and not f.inconclusive:
            parts.append(f"<p><b>Confidence:</b> {f.confidence_label}</p>")
            if f.very_low:
                parts.append("<p><span style=\"color:" + C["amber"] + "\"><b>Very low lead.</b></span></p>")
        if f.inconclusive:
            parts.append("<p><span style=\"color:" + C["amber"] + "\"><b>Inconclusive.</b></span> "
                         + html.escape(f.caveat) + "</p>")
        if not f.detected and f.status == "ok" and not f.inconclusive:
            parts.append("<p class='muted'>Information only: the probe ran and found nothing notable. "
                         "This is not a lead, and it is not proof that nothing is there.</p>")
        if f.summary and " • " in f.summary:
            parts.append("<ul>" + "".join(f"<li>{_rich(seg)}</li>" for seg in f.summary.split(" • ")) + "</ul>")
        else:
            parts.append(f"<p>{_rich(f.summary) if f.summary else html.escape(f.detail or 'No details.')}</p>")
        if f.regions and self.preview.can_play():
            items = "".join(
                f'<li><a href="mitraav-play:{t:.2f}">▶ {html.escape(label)}'
                f' (from {fmt_time(t)})</a></li>' for t, label in merge_play_links(f.regions)[:8])
            parts.append(f"<p><b>Play the source:</b></p><ul>{items}</ul>")
        if f.constrains:
            parts.append(f"<p><b>Helps with:</b> {html.escape(', '.join(f.constrains))}</p>")
        if f.follow_ups:
            parts.append("<p><b>Where to look next:</b></p><ul>")
            for u in f.follow_ups:
                name = html.escape(u.name)
                link = f'<a href="{html.escape(u.url, quote=True)}">{name}</a>' if u.url else name
                note = f" - {html.escape(u.note)}" if u.note else ""
                parts.append(f"<li>{link}{note}</li>")
            parts.append("</ul>")
        if f.detected and f.status == "ok" and not f.quiet:
            parts.append("<p class='muted'>A lead, not a conclusion: classifiers and heuristics mislabel "
                         "confusable, distant or stylised material. Fainter guesses are shown muted, not hidden.</p>")
        self.detail.setHtml("".join(parts))
