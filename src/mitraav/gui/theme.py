from __future__ import annotations

from string import Template

COLORS = dict(
    bg="#111214", panel="#1a1b1e", panel2="#242529", line="#35373c",
    text="#e8e8ea", muted="#8e9096", accent="#d6d6d9", accent_hi="#ffffff",
    green="#34d399", cyan="#22d3ee", amber="#fbbf24", red="#f87171",
    violet="#a78bfa", blue="#60a5fa",     # violet: finished, nothing found. blue: information
)

QSS = Template("""
QWidget { background: $bg; color: $text; font-size: 13px; }
QLabel { background: transparent; }
QLabel#title { font-size: 20px; font-weight: 800; letter-spacing: 0.5px; }
QLabel#subtitle, QLabel#hint { color: $muted; }
QLabel#chip { background: $panel2; border: 1px solid $line; border-radius: 10px;
              padding: 3px 10px; color: $muted; font-weight: 600; }
QLabel#chip[lead="true"] { color: $text; border-color: $muted; }
QFrame#card { background: $panel; border: 1px solid $line; border-radius: 14px; }
QLabel#dropzone { background: $panel; border: 2px dashed $line; border-radius: 14px;
                  padding: 16px; font-size: 14px; font-weight: 600; color: $muted; }
QLabel#dropzone[loaded="true"] { color: $text; border-style: solid; }
QLabel#dropzone[active="true"] { border: 2px solid $accent; background: $panel2; color: $text; }
QPushButton { background: $accent; color: $bg; border: none; border-radius: 9px;
              padding: 9px 18px; font-weight: 700; }
QPushButton:hover { background: $accent_hi; }
QPushButton[flat="true"], QPushButton#secondary { background: $panel2; color: $text;
              border: 1px solid $line; }
QPushButton#secondary:hover { border-color: $accent_hi; }
QPushButton:disabled { background: $panel; color: $muted; border: 1px solid $line; }
QTreeWidget, QListWidget { background: $panel; border: 1px solid $line; border-radius: 14px; outline: none;
               alternate-background-color: $panel; selection-background-color: $panel2; selection-color: $text; }
QTreeWidget::item { padding: 6px 8px; border-bottom: 1px solid $line; }
QTreeWidget::item:selected, QListWidget::item:selected { background: $panel2; }
QListWidget::item { padding: 8px; border-bottom: 1px solid $line; }
QListWidget::item:disabled { color: $muted; }
QProgressBar { background: $panel2; border: none; border-radius: 3px; max-height: 6px; color: transparent; }
QProgressBar::chunk { background: $blue; border-radius: 3px; }
QTableWidget { background: $panel; alternate-background-color: $panel; border: 1px solid $line;
               border-radius: 14px; gridline-color: transparent; outline: none;
               selection-background-color: $panel2; selection-color: $text; }
QTableWidget::item { padding: 6px 8px; border-bottom: 1px solid $line; }
QTableWidget::item:selected { background: $panel2; }
QHeaderView::section { background: $panel; color: $muted; border: none;
                       border-bottom: 1px solid $line; padding: 9px 8px; font-weight: 700;
                       font-size: 11px; text-transform: uppercase; }
QTextBrowser { background: $panel; border: 1px solid $line; border-radius: 14px; padding: 14px; }
QStatusBar { background: $bg; color: $muted; }
QComboBox { background: $panel2; border: 1px solid $line; border-radius: 8px; padding: 6px 12px; }
QComboBox QAbstractItemView { background: $panel; selection-background-color: $line; }
QSplitter::handle { background: transparent; width: 10px; height: 10px; }
QScrollBar:vertical, QScrollBar:horizontal { background: transparent; border: none; margin: 2px; }
QScrollBar:vertical { width: 10px; }
QScrollBar:horizontal { height: 10px; }
QScrollBar::handle { background: $line; border-radius: 4px; min-height: 28px; min-width: 28px; }
QScrollBar::handle:hover { background: $muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
""").substitute(COLORS)

DETAIL_CSS = Template(
    "h3 { color: $text; font-size: 17px; } p { line-height: 150%; } "
    "a { color: $accent_hi; text-decoration: underline; } b { color: $text; } "
    "li { margin-bottom: 6px; } .muted { color: $muted; } b { color: $text; }"
).substitute(COLORS)
