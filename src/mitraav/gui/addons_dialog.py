from __future__ import annotations

import subprocess

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core import addons
from . import theme

C = theme.COLORS
STATUS_COLOR = {"installed": C["green"], "partly installed": C["amber"], "not installed": C["muted"]}


class InstallWorker(QThread):
    line = Signal(str)
    finished_with = Signal(bool, str)          # ok, message

    def __init__(self, addon: addons.Addon):
        super().__init__()
        self.addon = addon
        self._stop = False
        self._proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        self._stop = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run(self) -> None:
        try:
            addons.run_steps(self.addon, self.line.emit, lambda: self._stop,
                             lambda p: setattr(self, "_proc", p))
            self.finished_with.emit(True, "")
        except addons.Cancelled:
            self.finished_with.emit(False, "Cancelled")
        except Exception as e:
            self.finished_with.emit(False, str(e))


class AddonsDialog(QDialog):
    """Optional probes: see what is installed, install what you want. Nothing installs by itself."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add-ons")
        self.resize(760, 680)
        self.worker: InstallWorker | None = None
        self._busy = False
        self.rows: dict[str, tuple[QLabel, QPushButton]] = {}

        intro = QLabel("Add-ons switch on extra probes. Installing one downloads files from the "
                       "internet once; scans themselves never connect to anything. Nothing here "
                       "is installed unless you click Install.")
        intro.setObjectName("hint")
        intro.setWordWrap(True)

        body = QWidget()
        col = QVBoxLayout(body)
        col.setSpacing(10)
        for a in addons.registry():
            col.addWidget(self._card(a))
        tools = QFrame()
        tools.setObjectName("card")
        tl = QVBoxLayout(tools)
        head = QLabel("System tools")
        head.setStyleSheet("font-weight: 700;")
        tl.addWidget(head)
        note = QLabel("These come from your operating system's package manager, so they can't be "
                      "installed from here.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        tl.addWidget(note)
        for tool, found, how in addons.system_tools():
            r = QLabel(f"<b>{tool}</b> &nbsp; "
                       + (f'<span style="color:{C["green"]}">found</span>' if found else
                          f'<span style="color:{C["amber"]}">missing</span> &nbsp; '
                          f'<span style="color:{C["muted"]}">{how}</span>'))
            r.setTextInteractionFlags(Qt.TextSelectableByMouse)
            r.setWordWrap(True)
            tl.addWidget(r)
        col.addWidget(tools)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setFrameShape(QFrame.NoFrame)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setVisible(False)
        self.log.setMinimumHeight(130)
        self.cancel_btn = QPushButton("Cancel install")
        self.cancel_btn.setObjectName("secondary")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.close_btn = QPushButton("Close")
        self.close_btn.setObjectName("secondary")
        self.close_btn.clicked.connect(self.accept)
        plugins_btn = QPushButton("Plugin probes...")
        plugins_btn.setObjectName("secondary")
        plugins_btn.setToolTip("Write or add your own probes (Python files); nothing runs until you enable it")
        plugins_btn.clicked.connect(self.open_plugins)
        row = QHBoxLayout()
        row.addWidget(plugins_btn)
        row.addStretch(1)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.log)
        lay.addLayout(row)
        self.refresh()

    def open_plugins(self) -> None:
        from .plugins_dialog import PluginsDialog
        PluginsDialog(self).exec()

    def _card(self, a: addons.Addon) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QHBoxLayout(card)
        text = QVBoxLayout()
        title = QLabel(f"<b>{a.name}</b> &nbsp; <span style='color:{C['muted']}'>{a.size}</span>"
                       + (f" &nbsp; <span style='color:{C['amber']}'>heavy</span>" if a.heavy else ""))
        desc = QLabel(a.adds)
        desc.setObjectName("hint")
        desc.setWordWrap(True)
        status = QLabel()
        text.addWidget(title)
        text.addWidget(desc)
        text.addWidget(status)
        btn = QPushButton("Install")
        btn.clicked.connect(lambda _=False, key=a.key: self.install(key))
        lay.addLayout(text, 1)
        lay.addWidget(btn, 0, Qt.AlignVCenter)
        self.rows[a.key] = (status, btn)
        return card

    def refresh(self) -> None:
        busy = self._busy
        for a in addons.registry():
            status, btn = self.rows[a.key]
            st = a.status()
            status.setText(f"<span style='color:{STATUS_COLOR[st]}'>{st}</span>")
            btn.setText("Finish install" if st == "partly installed" else "Install")
            btn.setEnabled(st != "installed" and not busy)
            btn.setVisible(st != "installed")
        self.close_btn.setEnabled(not busy)

    def install(self, key: str) -> None:
        a = addons.by_key(key)
        if a is None or self._busy:
            return
        steps = "\n".join(f"  • {s.label}" for s in a.pending())
        ok = QMessageBox.question(
            self, f"Install {a.name}?",
            f"This will download from the internet ({a.size}):\n\n{steps}\n\n"
            + ("This is a large download. " if a.heavy else "")
            + "Nothing else is installed. Continue?")
        if ok != QMessageBox.Yes:
            return
        self.log.clear()
        self.log.setVisible(True)
        self.cancel_btn.setVisible(True)
        self._busy = True
        self.worker = InstallWorker(a)
        self.worker.line.connect(lambda t: self.log.appendPlainText(t))
        self.worker.finished_with.connect(self._finished)
        self.worker.start()
        self.refresh()

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()

    def _finished(self, ok: bool, msg: str) -> None:
        self._busy = False                      # the thread may still be winding down; we are done
        self.cancel_btn.setVisible(False)
        if not ok:
            self.log.appendPlainText(f"Stopped: {msg}")
        self.refresh()

    def reject(self) -> None:
        if self._busy:
            return                                  # finish or cancel the install first
        super().reject()
