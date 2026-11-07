from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QThread, QTimer, Signal
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
from ..core.finding import Finding, MediaType, plain
from ..core.runner import media_of, planned, scan
from . import theme
from .addons_dialog import AddonsDialog
from .history_dialog import HistoryDialog
from .preview import MediaPreview
from .spectro_view import SpectrogramDialog, fmt_time

C = theme.COLORS
COLUMNS = ["Probe", "Result", "Confidence", "Could narrow", ""]
FOCUS_COL = 4
ROLE = Qt.UserRole
NAME_ROLE = Qt.UserRole + 1
LAST_GROUP = "File and metadata"

COLUMN_TIPS = [
    "The probe that ran.",
    "lead = the probe found something. info = it ran and found nothing notable (click for details; "
    "this is not proof of absence).",
    "How clearly the signal is present: clear, likely or faint. This is not how useful it is.",
    "How much a lead would cut down where or what to look at. 'file history only' means it "
    "describes how the file was made, not where it came from.",
    "Details: open the spectrogram on this finding's signal.",
]
EXPLAIN = ("<b>Confidence</b> is how clearly the signal is present. <b>Could narrow</b> is how much "
           "it would cut down where or what to look at. A clear signal can still narrow little.")


class ScanWorker(QThread):
    started_probe = Signal(str)
    found = Signal(object)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self) -> None:
        scan(self.path, on_result=self.found.emit, on_start=self.started_probe.emit)


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
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


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
    """A small 'Details' pill on rows that point at a signal in the recording."""

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
        p.drawText(r, Qt.AlignCenter, "Details")
        p.restore()


class ProbeTree(QTreeWidget):
    """The results tree; the pointer becomes a hand over a Details pill."""

    def mouseMoveEvent(self, e) -> None:
        idx = self.indexAt(e.position().toPoint())
        hand = idx.isValid() and idx.column() == FOCUS_COL and bool(idx.data(ROLE))
        self.viewport().setCursor(Qt.PointingHandCursor if hand else Qt.ArrowCursor)
        super().mouseMoveEvent(e)


class ConfidenceDelegate(QStyledItemDelegate):
    """A short bar for how clearly the signal is present, with its word beside it."""

    def paint(self, p, opt, idx) -> None:
        _base(self, p, opt, idx)
        strength = idx.data(ROLE)
        if strength is None:
            return
        col = C["green"] if strength >= 0.7 else C["cyan"] if strength >= 0.4 else C["amber"]
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(opt.rect).adjusted(8, 0, -58, 0)
        r.setTop(r.center().y() - 4)
        r.setHeight(8)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["panel2"]))
        p.drawRoundedRect(r, 4, 4)
        fill = QRectF(r)
        fill.setWidth(max(6.0, r.width() * min(1.0, strength)))
        p.setBrush(QColor(col))
        p.drawRoundedRect(fill, 4, 4)
        p.setPen(QColor(C["text"]))
        p.drawText(QRectF(opt.rect).adjusted(0, 0, -8, 0), Qt.AlignRight | Qt.AlignVCenter,
                   idx.data() or "")
        p.restore()


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
        self.plan: list[tuple[str, str]] = []        # (probe name, group) the scan will run
        self.running: str | None = None
        self.collapsed: set[str] = {"Visual context"}     # quiet context starts folded away
        self.worker: ScanWorker | None = None
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
        header.addLayout(split)
        header.addWidget(addons_btn)

        self.preview = MediaPreview()

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)

        self.focus_ok = False                            # audio or video: a signal can be shown
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
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setSectionResizeMode(FOCUS_COL, QHeaderView.Fixed)
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 140)
        self.tree.setColumnWidth(2, 112)
        self.tree.setColumnWidth(FOCUS_COL, 74)
        self.tree.setItemDelegateForColumn(FOCUS_COL, FocusDelegate(self.tree))
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemDoubleClicked.connect(self._item_double_clicked)
        self.tree.setItemDelegateForColumn(1, PillDelegate(self.tree))
        self.tree.setItemDelegateForColumn(2, ConfidenceDelegate(self.tree))
        self.tree.itemSelectionChanged.connect(self.show_detail)
        self.tree.itemCollapsed.connect(lambda it: self.collapsed.add(it.text(0)))
        self.tree.itemExpanded.connect(lambda it: self.collapsed.discard(it.text(0)))

        self.detail = QTextBrowser()
        self.detail.setOpenLinks(False)                 # we route links: play-from-here or the browser
        self.detail.anchorClicked.connect(self._link_clicked)
        self.detail.document().setDefaultStyleSheet(theme.DETAIL_CSS)
        self.detail.setHtml(self._placeholder())

        right = QWidget()                                # preview above the details
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        rl.addWidget(self.preview)
        rl.addWidget(self.detail, 1)

        split = QSplitter(Qt.Horizontal)
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
        return ('<p class="muted">Select a probe on the left to see what it found, what it could '
                f'narrow, and where to look next.</p><p class="muted">{EXPLAIN}</p>')

    # --- input ---
    def _drop_active(self, on: bool) -> None:
        self.drop.setProperty("active", on)
        self.drop.style().unpolish(self.drop)
        self.drop.style().polish(self.drop)

    def choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open file to scan")
        if name:
            self.start_scan(Path(name))

    def open_addons(self) -> None:
        AddonsDialog(self).exec()

    def open_history(self) -> None:
        dlg = HistoryDialog(self)
        if dlg.exec() and dlg.selected_path:
            self.start_scan(dlg.selected_path)

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
            self.start_scan(Path(urls[0]))

    # --- scanning ---
    def start_scan(self, path: Path) -> None:
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Scan already running, please wait")
            return
        if not path.is_file():
            self.statusBar().showMessage(f"Not a file: {path}")
            return
        self.findings.clear()
        self.by_name.clear()
        self.running = None
        self.plan = [(cls.name, cls.group) for cls, _ in planned(path)]
        self.path = path
        self.drop.setFullText(path.name)
        self.drop.setProperty("loaded", True)
        self._drop_active(False)
        self.drop.setToolTip(str(path))
        self.preview.load(path, media_of(path))
        self.detail.setHtml(self._placeholder())
        self.focus_ok = media_of(path) in (MediaType.AUDIO, MediaType.VIDEO)
        if not self.plan:
            self.refresh()
            self.statusBar().showMessage("No probes apply to this file type yet")
            return
        self.progress.setRange(0, len(self.plan))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.statusBar().showMessage("Scanning...")
        self.refresh()
        self.timer.start()
        self.worker = ScanWorker(path)
        self.worker.started_probe.connect(self.on_started)
        self.worker.found.connect(self.add_finding)
        self.worker.finished.connect(self.scan_done)
        self.worker.start()

    def on_started(self, name: str) -> None:
        self.running = name
        done = len(self.findings)
        self.statusBar().showMessage(f"Running {name} ({done + 1} of {len(self.plan)})...")
        self.refresh()

    def add_finding(self, f: Finding) -> None:
        self.findings.append(f)
        self.by_name[f.probe] = f
        if self.running == f.probe:
            self.running = None
        if self.plan:
            self.progress.setValue(min(len(self.findings), len(self.plan)))
        self.refresh()

    def scan_done(self) -> None:
        self.timer.stop()
        self.running = None
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
        if column == FOCUS_COL and item.data(FOCUS_COL, ROLE):
            f = self.by_name.get(item.data(0, NAME_ROLE))
            if f is not None:
                self.open_spectrogram(f.regions)

    def _item_double_clicked(self, item, column: int) -> None:
        if column != FOCUS_COL and item.data(FOCUS_COL, ROLE):       # double-click anywhere on the row
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
        if not hasattr(self, "_detail_windows"):
            self._detail_windows = []
        self._detail_windows.append(dlg)                   # closed windows delete themselves: drop them on destroy
        dlg.destroyed.connect(lambda _=None, d=dlg: self._detail_windows.remove(d) if d in self._detail_windows else None)
        dlg.show()

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
        for g, items in rows.items():
            items.sort(key=lambda r: (r[1] is None, -(r[1].score if r[1] else 0),
                                      r[1].status != "ok" if r[1] else False, r[0]))

        def key(g: str):
            best = max((f.score for _, f in rows[g] if f and not f.quiet), default=0.0)
            all_quiet = all(f is not None and f.quiet for _, f in rows[g]) and bool(rows[g])
            return (g == LAST_GROUP, all_quiet, -best, g)
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
                                  if all(f for _, f in items) else f"{len(items)} probes"), "", ""])
            font = QFont(self.font())
            font.setBold(True)
            g.setFont(0, font)
            g.setFlags(Qt.ItemIsEnabled)
            for c in range(len(COLUMNS)):
                g.setBackground(c, QColor(C["panel2"]))
            g.setForeground(1, QColor(C["muted"]))
            self.tree.addTopLevelItem(g)
            for name, f in items:
                it = QTreeWidgetItem([name.replace("_", " "), "", "", "", ""])
                it.setData(0, NAME_ROLE, name)
                if f is None:
                    running = name == self.running
                    it.setText(1, f"running{dots}" if running else "queued")
                    it.setData(1, ROLE, C["text"] if running else C["muted"])
                    it.setForeground(0, QColor(C["text"] if running else C["muted"]))
                else:
                    label, color = _badge(f)
                    it.setText(1, label)
                    it.setData(1, ROLE, color)
                    it.setToolTip(0, plain(f.summary) or f.detail)
                    if f.regions and self.focus_ok and f.status == "ok":
                        it.setData(FOCUS_COL, ROLE, True)
                        it.setToolTip(FOCUS_COL, "Open the spectrogram on this finding's signal")
                    if f.inconclusive:
                        it.setText(3, "busy audio")
                    elif f.detected and f.status == "ok" and f.quiet:
                        it.setText(3, "visible to the eye")
                    elif f.detected and f.status == "ok":
                        it.setText(2, f.confidence_label)
                        it.setData(2, ROLE, f.strength)
                        it.setText(3, f.narrows_label)
                    it.setForeground(3, QColor(C["muted"]))
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
        parts = [f"<h3>{html.escape(f.probe)} "
                 f'<span style="color:{color}">{html.escape(label)}</span></h3>']
        if f.detected and f.status == "ok" and f.quiet and not f.inconclusive:
            parts.append("<p class='muted'>Context only: this reports what the picture shows to the "
                         "eye, so it is not counted as a lead.</p>")
        elif f.detected and f.status == "ok" and not f.inconclusive:
            parts.append(f"<p><b>Confidence:</b> {f.confidence_label} &nbsp; "
                         f"<b>Could narrow:</b> {f.narrows_label}</p>")
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
                f'<li><a href="mitraav-play:{_play_at(r):.2f}">▶ {html.escape(r["label"])}'
                f' (from {fmt_time(_play_at(r))})</a></li>' for r in f.regions[:5])
            parts.append(f"<p><b>Play the source:</b></p><ul>{items}</ul>")
        if f.constrains:
            parts.append(f"<p><b>Helps with:</b> {html.escape(', '.join(f.constrains))}</p>")
        if f.follow_ups:
            parts.append("<p><b>Follow up with:</b></p><ul>")
            for u in f.follow_ups:
                name = html.escape(u.name)
                link = f'<a href="{html.escape(u.url, quote=True)}">{name}</a>' if u.url else name
                note = f" - {html.escape(u.note)}" if u.note else ""
                parts.append(f"<li>{link}{note}</li>")
            parts.append("</ul>")
        self.detail.setHtml("".join(parts))
