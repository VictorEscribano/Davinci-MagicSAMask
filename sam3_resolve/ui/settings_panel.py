"""
Settings slide-in panel (Step 14).

Slides in from the right over the right panel via QPropertyAnimation.
Sections:
  Model & device  — model selector, device combo, fp16 toggle
  Proxy & cache   — quality preset, cache size label, clear button
  Export          — output folder, bit depth info, worker count
  Keybindings     — 2-column QTableWidget with QKeySequenceEdit cells
  Repair          — re-runs SetupWizard steps that are failing

The panel persists settings to Config on Apply; no writes on Cancel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sam3_resolve.config import Config
from sam3_resolve.constants import (
    COLOR_ACCENT,
    COLOR_TEXT_SECONDARY,
    EXPORT_WORKERS,
    PROXY_CRF,
)

logger = logging.getLogger(__name__)

ANIMATION_MS = 220

# Default keybindings: (action_label, default_shortcut)
DEFAULT_KEYBINDINGS: list[tuple[str, str]] = [
    ("Add positive point",  ""),          # left-click — no shortcut
    ("Add negative point",  ""),          # right-click
    ("Undo last prompt",    "Ctrl+Z"),
    ("Redo",                "Ctrl+Shift+Z"),
    ("Clear all prompts",   "Ctrl+Shift+C"),
    ("Start propagation",   "Ctrl+Return"),
    ("Switch to Points",    "P"),
    ("Switch to Box",       "B"),
    ("Switch to Mask",      "M"),
    ("Toggle fullscreen",   "F"),
    ("Zoom in",             "="),
    ("Zoom out",            "-"),
    ("Reset zoom",          "0"),
]


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; "
        "font-weight: 700; letter-spacing: 1px;"
    )
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #3A3A3A;")
    return line


# ── Settings Panel ─────────────────────────────────────────────────────────

class SettingsPanel(QWidget):
    """
    Slide-in settings overlay.

    Usage::

        panel = SettingsPanel(parent=right_panel_widget)
        panel.show_panel()   # slides in from right
        panel.hide_panel()   # slides back out

    Signals:
        settings_applied():  user clicked Apply — Config was updated
        repair_requested():  user clicked Repair button
    """

    settings_applied = pyqtSignal()
    repair_requested = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("settings_panel")
        self.setStyleSheet(
            "QWidget#settings_panel { background-color: #1E1E1E; "
            "border-left: 1px solid #3A3A3A; }"
        )
        self.setFixedWidth(300)
        self._visible = False
        self._anim: Optional[QPropertyAnimation] = None
        self._build()
        # Hide until explicitly opened; geometry is computed on show_panel()
        self.hide()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("background-color: #2A2A2A; border-bottom: 1px solid #3A3A3A;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 8, 0)
        h_layout.addWidget(QLabel("Settings"))
        h_layout.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setObjectName("toolbar_btn")
        close_btn.clicked.connect(self.hide_panel)
        h_layout.addWidget(close_btn)
        outer.addWidget(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self._build_model_section(layout)
        layout.addWidget(_separator())
        self._build_resolve_path_section(layout)
        layout.addWidget(_separator())
        self._build_proxy_section(layout)
        layout.addWidget(_separator())
        self._build_export_section(layout)
        layout.addWidget(_separator())
        self._build_keybindings_section(layout)
        layout.addWidget(_separator())
        self._build_repair_section(layout)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # Apply / Cancel buttons
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(8, 6, 8, 8)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("btn_cancel")
        btn_cancel.clicked.connect(self.hide_panel)
        btn_apply = QPushButton("Apply")
        btn_apply.setObjectName("btn_accept")
        btn_apply.clicked.connect(self._apply)
        btn_bar.addWidget(btn_cancel)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_apply)
        outer.addLayout(btn_bar)

    def _build_model_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("Model & Device"))
        form = QFormLayout()
        form.setSpacing(6)

        self._model_combo = QComboBox()
        self._model_combo.addItems(["SAM3 Large (2.4 GB)", "SAM3 Base (360 MB)"])
        cfg = Config.instance()
        model = cfg.get("model_name", "large")
        self._model_combo.setCurrentIndex(0 if model == "large" else 1)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["Auto-detect", "CUDA", "MPS", "CPU"])
        device = cfg.get("device", "auto")
        idx = {"auto": 0, "cuda": 1, "mps": 2, "cpu": 3}.get(device.lower(), 0)
        self._device_combo.setCurrentIndex(idx)

        self._fp16_check = QCheckBox("Use float16 (faster, less VRAM)")
        self._fp16_check.setChecked(cfg.get("use_fp16", True))

        form.addRow("Model:", self._model_combo)
        form.addRow("Device:", self._device_combo)
        form.addRow("", self._fp16_check)
        layout.addLayout(form)

        restart_lbl = QLabel("⚠ Model/device changes require restart")
        restart_lbl.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")
        layout.addWidget(restart_lbl)

    def _build_resolve_path_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("DaVinci Resolve"))
        form = QFormLayout()
        form.setSpacing(6)

        cfg = Config.instance()
        self._resolve_path_edit = QLineEdit(cfg.get("resolve_api_path", ""))
        self._resolve_path_edit.setPlaceholderText("(auto-detected)")
        self._resolve_path_edit.setToolTip(
            "Path to the folder containing fusionscript.so / fusionscript.dll.\n"
            "Leave blank to auto-detect. Set RESOLVE_INSTALL_DIR env var for\n"
            "non-standard installs (Resolve sets this itself for its own scripts)."
        )
        form.addRow("API path:", self._resolve_path_edit)
        layout.addLayout(form)

        hint = QLabel(
            "Tip: set RESOLVE_INSTALL_DIR=/your/resolve/dir before launching\n"
            "the plugin, or paste the path above and click Apply."
        )
        hint.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _build_proxy_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("Proxy & Cache"))
        form = QFormLayout()
        form.setSpacing(6)

        cfg = Config.instance()
        self._proxy_crf_spin = QSpinBox()
        self._proxy_crf_spin.setRange(1, 51)
        self._proxy_crf_spin.setValue(cfg.get("proxy_crf", PROXY_CRF))
        self._proxy_crf_spin.setToolTip("Lower = better quality, larger file")

        cache_path = str(Path.home() / ".sam3_resolve_cache")
        cache_lbl = QLabel(cache_path)
        cache_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px;")
        cache_lbl.setWordWrap(True)

        self._clear_cache_btn = QPushButton("Clear frame cache")
        self._clear_cache_btn.clicked.connect(self._clear_cache)

        form.addRow("Proxy CRF:", self._proxy_crf_spin)
        form.addRow("Cache:", cache_lbl)
        layout.addLayout(form)
        layout.addWidget(self._clear_cache_btn)

    def _build_export_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("Export"))
        form = QFormLayout()
        form.setSpacing(6)

        cfg = Config.instance()
        self._export_dir_edit = QLineEdit(cfg.get("export_dir", ""))
        self._export_dir_edit.setPlaceholderText("(same folder as source clip)")

        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 8)
        self._workers_spin.setValue(cfg.get("export_workers", EXPORT_WORKERS))

        depth_lbl = QLabel("16-bit single-channel PNG (fixed for Resolve compatibility)")
        depth_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px;")
        depth_lbl.setWordWrap(True)

        form.addRow("Output dir:", self._export_dir_edit)
        form.addRow("Workers:", self._workers_spin)
        form.addRow("Bit depth:", depth_lbl)
        layout.addLayout(form)

    def _build_keybindings_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("Keybindings"))
        self._keybind_table = QTableWidget(len(DEFAULT_KEYBINDINGS), 2)
        self._keybind_table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._keybind_table.horizontalHeader().setStretchLastSection(True)
        self._keybind_table.verticalHeader().setVisible(False)
        self._keybind_table.setFixedHeight(200)
        self._keybind_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        cfg = Config.instance()
        keybindings_cfg: dict = cfg.get("keybindings", {})

        for row, (action, default_seq) in enumerate(DEFAULT_KEYBINDINGS):
            action_item = QTableWidgetItem(action)
            action_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._keybind_table.setItem(row, 0, action_item)

            seq_str = keybindings_cfg.get(action, default_seq)
            seq_edit = QKeySequenceEdit(QKeySequence(seq_str))
            self._keybind_table.setCellWidget(row, 1, seq_edit)

        layout.addWidget(self._keybind_table)

    def _build_repair_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_section_label("Repair"))
        lbl = QLabel("Re-run setup checks to fix missing dependencies or model files.")
        lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        repair_btn = QPushButton("⟳  Open Setup Wizard")
        repair_btn.clicked.connect(self.repair_requested)
        layout.addWidget(repair_btn)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _apply(self) -> None:
        cfg = Config.instance()
        cfg.set("model_name", "large" if self._model_combo.currentIndex() == 0 else "base")
        device_map = {0: "auto", 1: "cuda", 2: "mps", 3: "cpu"}
        cfg.set("device", device_map.get(self._device_combo.currentIndex(), "auto"))
        cfg.set("use_fp16", self._fp16_check.isChecked())
        cfg.set("resolve_api_path", self._resolve_path_edit.text().strip())
        cfg.set("proxy_crf", self._proxy_crf_spin.value())
        cfg.set("export_dir", self._export_dir_edit.text().strip())
        cfg.set("export_workers", self._workers_spin.value())

        keybindings: dict[str, str] = {}
        for row, (action, _default) in enumerate(DEFAULT_KEYBINDINGS):
            widget = self._keybind_table.cellWidget(row, 1)
            if isinstance(widget, QKeySequenceEdit):
                keybindings[action] = widget.keySequence().toString()
        cfg.set("keybindings", keybindings)

        cfg.save()
        self.settings_applied.emit()
        self.hide_panel()

    def _clear_cache(self) -> None:
        import shutil
        cache_dir = Path.home() / ".sam3_resolve_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Frame cache cleared: %s", cache_dir)

    # ── Animation ──────────────────────────────────────────────────────────

    def show_panel(self) -> None:
        if self._visible:
            return
        self._visible = True
        # Set starting position (off-screen right) now that parent has final size
        self.setGeometry(self._hidden_rect())
        self.show()
        self.raise_()
        self._animate_to(self._shown_rect())

    def hide_panel(self) -> None:
        if not self._visible:
            return
        self._visible = False
        self._animate_to(self._hidden_rect(), on_done=self.hide)

    def _animate_to(self, target: QRect, on_done=None) -> None:
        if self._anim and self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()
        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(ANIMATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(self.geometry())
        self._anim.setEndValue(target)
        if on_done:
            self._anim.finished.connect(on_done)
        self._anim.start()

    def _shown_rect(self) -> QRect:
        p = self.parent()
        if p is None:
            return QRect(0, 0, self.width(), self.height())
        pw, ph = p.width(), p.height()
        return QRect(pw - self.width(), 0, self.width(), ph)

    def _hidden_rect(self) -> QRect:
        p = self.parent()
        if p is None:
            return QRect(self.width(), 0, self.width(), self.height())
        pw, ph = p.width(), p.height()
        return QRect(pw, 0, self.width(), ph)

    def _move_to_hidden(self) -> None:
        self.setGeometry(self._hidden_rect())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)

    def parentResized(self) -> None:
        """Call this when the parent widget is resized to keep the panel anchored."""
        if self._visible:
            self.setGeometry(self._shown_rect())
        else:
            self.hide()

    # ── Accessors (for tests) ──────────────────────────────────────────────

    @property
    def is_panel_visible(self) -> bool:
        return self._visible

    def get_keybinding(self, action: str) -> str:
        for row, (act, _) in enumerate(DEFAULT_KEYBINDINGS):
            if act == action:
                widget = self._keybind_table.cellWidget(row, 1)
                if isinstance(widget, QKeySequenceEdit):
                    return widget.keySequence().toString()
        return ""
