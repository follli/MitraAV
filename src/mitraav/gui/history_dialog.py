from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..core import history


class HistoryDialog(QDialog):
    """Previously scanned files, newest first. Pick one and scan it again."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan history")
        self.resize(560, 420)
        self.selected_path: Path | None = None
        self.hint = QLabel("Double-click a file to scan it again. Only names, times and counts "
                           "are kept, never the results.")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _: self.scan_again())
        self.scan_btn = QPushButton("Scan again")
        self.scan_btn.clicked.connect(self.scan_again)
        self.clear_btn = QPushButton("Clear history")
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self.clear)
        close = QPushButton("Close")
        close.setObjectName("secondary")
        close.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.scan_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.hint)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        self.fill()

    def fill(self) -> None:
        self.list.clear()
        entries = history.load()
        for e in entries:
            exists = Path(e["path"]).exists()
            n = e.get("leads", 0)
            item = QListWidgetItem(f"{e['name']}    {n} lead{'s' if n != 1 else ''}  ·  "
                                   f"{e.get('when', '').replace('T', ' ')}")
            item.setToolTip(e["path"] if exists else e["path"] + "  (file no longer there)")
            item.setData(Qt.UserRole, e["path"])
            if not exists:
                item.setFlags(Qt.NoItemFlags)
            self.list.addItem(item)
        if not entries:
            empty = QListWidgetItem("No scans yet")
            empty.setFlags(Qt.NoItemFlags)
            self.list.addItem(empty)
        self.scan_btn.setEnabled(bool(entries))
        self.clear_btn.setEnabled(bool(entries))
        if entries:
            self.list.setCurrentRow(0)

    def scan_again(self) -> None:
        item = self.list.currentItem()
        if item and item.data(Qt.UserRole):
            self.selected_path = Path(item.data(Qt.UserRole))
            self.accept()

    def clear(self) -> None:
        ok = QMessageBox.question(self, "Clear history", "Remove the list of scanned files?")
        if ok == QMessageBox.Yes:
            history.clear()
            self.fill()
