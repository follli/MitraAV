from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from . import theme
from .main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("MitraAV")
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.start_scan(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
