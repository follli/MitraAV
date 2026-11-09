"""Plugin probes: see the plugin files in your plugins folder, read what they claim to be, and
enable or disable them. Nothing runs until you enable it (and an edited file is disabled again)."""
from __future__ import annotations

import html
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core import plugins
from . import theme

C = theme.COLORS
STATUS = {"enabled": C["green"], "disabled": C["muted"], "changed": C["amber"]}
TEMPLATE = Path(plugins.__file__).parent.parent / "templates" / "example_plugin.py"


class PluginsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plugin probes")
        self.resize(720, 560)
        lay = QVBoxLayout(self)
        intro = QLabel(
            "A plugin is a Python file that adds a probe (see docs/PLUGINS.md). <b>A plugin is code that "
            "runs with your permissions</b>, so nothing loads until you enable it, and a file you edit "
            "afterwards is switched off again until you re-enable it. Only enable code you have read or "
            "trust. Changes apply the next time you open a file.")
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        lay.addWidget(intro)
        self.folder = plugins.plugin_dirs()[0]
        self.where = QLabel(f"Folder: {self.folder}")
        self.where.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.where)
        bar = QHBoxLayout()
        for text, fn in (("Open folder", self.open_folder), ("Add example plugin", self.add_example),
                         ("Refresh", self.refresh)):
            b = QPushButton(text)
            b.setObjectName("secondary")
            b.clicked.connect(lambda _=False, f=fn: f())
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.body = QWidget()
        self.col = QVBoxLayout(self.body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.body)
        lay.addWidget(scroll, 1)
        close = QPushButton("Close")
        close.setObjectName("secondary")
        close.clicked.connect(self.accept)
        lay.addWidget(close, 0, Qt.AlignRight)
        self.refresh()

    # --- actions ---
    def open_folder(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))

    def add_example(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        dest = self.folder / "loudest_moment.py"
        n = 2
        while dest.exists():
            dest = self.folder / f"loudest_moment_{n}.py"
            n += 1
        shutil.copy(TEMPLATE, dest)
        self.refresh()

    def toggle(self, pf: plugins.PluginFile) -> None:
        if pf.status == "enabled":
            plugins.disable(pf.path)
        else:
            ok = QMessageBox.question(
                self, f"Enable {pf.path.name}?",
                f"{pf.path.name} is Python code and will run on your computer with your permissions "
                "whenever a scan uses it.\n\nEnable it only if you wrote it or have read it and trust "
                "its source. If you edit the file later it will be switched off again.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return
            plugins.enable(pf.path)
            plugins.load_enabled()                      # surfaces an import error straight away
        self.refresh()

    def refresh(self) -> None:
        while self.col.count():
            item = self.col.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        found = plugins.list_plugins()
        if not found:
            empty = QLabel("No plugin files yet. Use \"Add example plugin\" or put a .py file in the folder.")
            empty.setObjectName("hint")
            self.col.addWidget(empty)
        for pf in found:
            self.col.addWidget(self._row(pf))
        self.col.addStretch(1)

    def _row(self, pf: plugins.PluginFile) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QHBoxLayout(card)
        text = QVBoxLayout()
        err = plugins.ERRORS.get(str(pf.path.resolve()), "") or pf.error
        label = {"enabled": "enabled", "disabled": "off", "changed": "edited since enabled: off"}[pf.status]
        head = QLabel(f"<b>{html.escape(pf.path.name)}</b> &nbsp; "
                      f"<span style='color:{STATUS[pf.status]}'>{label}</span>")
        text.addWidget(head)
        if pf.doc:
            d = QLabel(html.escape(pf.doc))
            d.setObjectName("hint")
            d.setWordWrap(True)
            text.addWidget(d)
        if pf.probes:
            p = QLabel("Defines: " + ", ".join(pf.probes))
            p.setObjectName("hint")
            text.addWidget(p)
        if err:
            e = QLabel(f"<span style='color:{C['red']}'>{html.escape(err)}</span>")
            e.setWordWrap(True)
            text.addWidget(e)
        btn = QPushButton("Disable" if pf.enabled else "Enable")
        btn.setObjectName("secondary" if pf.enabled else "")
        btn.clicked.connect(lambda _=False, f=pf: self.toggle(f))
        lay.addLayout(text, 1)
        lay.addWidget(btn, 0, Qt.AlignVCenter)
        return card
