"""
Log panel — coloured, timestamped, scrolling log widget.

LogPanel is a thin wrapper around QPlainTextEdit that formats each entry
as HTML with a timestamp prefix and level-based colouring.  It is embedded
inside BottomPanel's tab widget and can also be used stand-alone in tests.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from PyQt6.QtWidgets import QPlainTextEdit, QWidget

logger = logging.getLogger(__name__)

# Level → hex colour
_LEVEL_COLORS: dict[str, str] = {
    "INFO":  "#909090",
    "OK":    "#5AB85A",
    "WARN":  "#E8A030",
    "ERROR": "#E85050",
    "DEBUG": "#6090C0",
}

MAX_LOG_LINES = 500


class LogPanel(QPlainTextEdit):
    """
    Scrolling, HTML-formatted log panel.

    Usage::

        panel = LogPanel()
        panel.append_log("OK",   "Model loaded")
        panel.append_log("WARN", "Low VRAM — switching to Base model")
        panel.append_log("ERROR", "File not found: foo.mov")
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("log_view")
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_LOG_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    # ── Public API ─────────────────────────────────────────────────────────

    def append_log(self, level: str, message: str) -> None:
        """Append a timestamped, colour-coded line."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = _LEVEL_COLORS.get(level.upper(), "#909090")
        html = (
            f'<span style="color:#555555">{ts}</span>&nbsp;'
            f'<span style="color:{color};font-weight:600">[{level}]</span>&nbsp;'
            f'<span style="color:#E8E8E8">{message}</span>'
        )
        self.appendHtml(html)
        # Auto-scroll to bottom
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self) -> None:
        self.clear()

    def log_lines(self) -> list[str]:
        """Return current plain-text lines (for tests / export)."""
        return self.toPlainText().splitlines()
