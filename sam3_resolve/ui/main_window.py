"""
SAM3 Resolve Plugin — Main Window.

Implements the exact panel layout from the spec mockup:
  Header (56px) · Left (260px) · Canvas (flex) · Right (240px) · Bottom (tabs)
All heavy work is delegated to QThread workers; this file is pure UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sam3_resolve.constants import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
    MAX_OBJECTS,
    PLUGIN_VERSION,
)
from sam3_resolve.core.resolve_bridge import ClipInfo, MockResolveBridge
from sam3_resolve.core.sam3_runner import MockSAM3Runner

logger = logging.getLogger(__name__)

# ── Style helpers ──────────────────────────────────────────────────────────

def _load_stylesheet() -> str:
    qss_path = Path(__file__).parent / "styles.qss"
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    logger.warning("styles.qss not found at %s", qss_path)
    return ""


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_label")
    return lbl


def _info_row(key: str, value: str = "—") -> tuple[QLabel, QLabel]:
    k = QLabel(key)
    k.setObjectName("info_key")
    v = QLabel(value)
    v.setObjectName("info_value")
    v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return k, v


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #3A3A3A;")
    line.setFixedHeight(1)
    return line


# ── Header bar ─────────────────────────────────────────────────────────────

class HeaderBar(QWidget):
    settings_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("header_bar")
        self.setFixedHeight(56)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # Left: title + version badge
        title = QLabel("SAM3 Mask Tracker")
        title.setObjectName("title_label")
        badge = QLabel(f"SAM3 v{PLUGIN_VERSION}")
        badge.setObjectName("version_badge")

        layout.addWidget(title)
        layout.addWidget(badge)
        layout.addSpacing(16)

        # Centre: clip info (filled by update_clip_info)
        centre = QVBoxLayout()
        centre.setSpacing(2)
        centre.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.clip_name_lbl = QLabel("No clip selected")
        self.clip_name_lbl.setObjectName("clip_name_label")
        self.clip_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.clip_meta_lbl = QLabel("—")
        self.clip_meta_lbl.setObjectName("clip_meta_label")
        self.clip_meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        centre.addWidget(self.clip_name_lbl)
        centre.addWidget(self.clip_meta_lbl)

        layout.addLayout(centre, stretch=1)

        # Right: GPU status + gear
        right = QHBoxLayout()
        right.setSpacing(6)
        right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.gpu_dot = QLabel("●")
        self.gpu_dot.setObjectName("gpu_dot")
        self.gpu_status = QLabel("Detecting GPU…")
        self.gpu_status.setObjectName("gpu_status_label")

        gear_btn = QPushButton("⚙")
        gear_btn.setObjectName("toolbar_btn")
        gear_btn.setToolTip("Settings")
        gear_btn.clicked.connect(self.settings_requested)

        right.addWidget(self.gpu_dot)
        right.addWidget(self.gpu_status)
        right.addSpacing(8)
        right.addWidget(gear_btn)
        layout.addLayout(right)

    def update_clip_info(self, clip: ClipInfo) -> None:
        self.clip_name_lbl.setText(clip.name)
        fps = f"{clip.fps:.3f}".rstrip("0").rstrip(".")
        dur = f"{clip.duration_seconds:.1f}s"
        self.clip_meta_lbl.setText(
            f"{clip.resolution_label}  ·  {fps} fps  ·  {dur}"
        )

    def update_gpu_status(self, label: str, ready: bool = True) -> None:
        self.gpu_dot.setStyleSheet(
            f"color: {'#5AB85A' if ready else '#E85050'};"
        )
        self.gpu_status.setStyleSheet(
            f"color: {'#5AB85A' if ready else '#E85050'}; font-size: 11px;"
        )
        self.gpu_status.setText(label)


# ── Left panel ─────────────────────────────────────────────────────────────

class LeftPanel(QWidget):
    prompt_mode_changed = pyqtSignal(str)   # 'points' | 'box' | 'mask' | 'text'
    proxy_source_changed = pyqtSignal(str)  # 'use_proxy' | 'generate' | 'full'
    proxy_preset_changed = pyqtSignal(str)
    estimate_refresh_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("left_panel")
        self.setFixedWidth(260)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)

        # ── Prompt Mode ──────────────────────────────────────
        layout.addWidget(_section_label("Prompt Mode"))

        self._mode_points = QRadioButton("Points")
        self._mode_points.setChecked(True)
        self._mode_box   = QRadioButton("Box")
        self._mode_mask  = QRadioButton("Mask")
        self._mode_text  = QRadioButton("Text")

        shortcut_map = {
            self._mode_points: "P",
            self._mode_box:    "B",
            self._mode_mask:   "M",
            self._mode_text:   "T",
        }

        mode_grid = QGridLayout()
        mode_grid.setHorizontalSpacing(0)
        mode_grid.setVerticalSpacing(2)
        for row, (rb, key) in enumerate(shortcut_map.items()):
            mode_grid.addWidget(rb, row, 0)
            sc = QLabel(key)
            sc.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sc.setStyleSheet("color: #555555; font-size: 11px;")
            mode_grid.addWidget(sc, row, 1)

        for rb in shortcut_map:
            rb.toggled.connect(self._on_mode_toggled)

        layout.addLayout(mode_grid)
        layout.addSpacing(8)

        # ── Proxy Source ─────────────────────────────────────
        layout.addWidget(_section_label("Proxy Source"))

        self._proxy_use     = QRadioButton("Use proxy")
        self._proxy_gen     = QRadioButton("Generate")
        self._proxy_full    = QRadioButton("Full resolution")
        self._proxy_use.setChecked(True)

        self._proxy_label   = QLabel("1/4 · 1920×1080")
        self._proxy_label.setStyleSheet("color: #909090; font-size: 10px; margin-left: 20px;")

        self._proxy_preset  = QComboBox()
        self._proxy_preset.addItems(["1/4 res — Recommended", "1/2 res — Balance", "Full resolution"])
        self._proxy_preset.setStyleSheet("margin-left: 20px;")
        self._proxy_preset.setEnabled(False)

        for rb in (self._proxy_use, self._proxy_gen, self._proxy_full):
            rb.toggled.connect(self._on_proxy_toggled)
        self._proxy_preset.currentIndexChanged.connect(
            lambda: self.proxy_preset_changed.emit(self._proxy_preset.currentText())
        )

        layout.addWidget(self._proxy_use)
        layout.addWidget(self._proxy_label)
        layout.addWidget(self._proxy_gen)
        layout.addWidget(self._proxy_preset)
        layout.addWidget(self._proxy_full)
        layout.addSpacing(8)

        # ── Clip Info ─────────────────────────────────────────
        layout.addWidget(_section_label("Clip Info"))

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(8)
        info_grid.setVerticalSpacing(3)
        info_grid.setColumnStretch(1, 1)

        self._info_fields: dict[str, QLabel] = {}
        for row, (key, attr) in enumerate([
            ("File", "file"), ("Path", "path"), ("Res", "res"),
            ("Duration", "duration"), ("FPS", "fps"),
            ("In", "in_pt"), ("Out", "out_pt"),
        ]):
            k, v = _info_row(key)
            info_grid.addWidget(k, row, 0)
            info_grid.addWidget(v, row, 1)
            self._info_fields[attr] = v

        layout.addLayout(info_grid)
        layout.addSpacing(8)

        # ── Estimated Time ────────────────────────────────────
        layout.addWidget(_section_label("Estimated"))

        est_grid = QGridLayout()
        est_grid.setHorizontalSpacing(8)
        est_grid.setVerticalSpacing(3)
        for row, (key, attr) in enumerate([
            ("Model", "est_model"), ("Proxy", "est_proxy"), ("Estimated", "est_time"),
        ]):
            k, v = _info_row(key)
            est_grid.addWidget(k, row, 0)
            est_grid.addWidget(v, row, 1)
            self._info_fields[attr] = v

        refresh_btn = QPushButton("↺ Refresh")
        refresh_btn.setObjectName("refresh_btn")
        refresh_btn.clicked.connect(self.estimate_refresh_requested)
        est_grid.addWidget(refresh_btn, 3, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)

        layout.addLayout(est_grid)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Signal handlers ────────────────────────────────────────────────────

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        mapping = {
            self._mode_points: "points",
            self._mode_box:    "box",
            self._mode_mask:   "mask",
            self._mode_text:   "text",
        }
        for rb, mode in mapping.items():
            if rb.isChecked():
                self.prompt_mode_changed.emit(mode)
                break

    def _on_proxy_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._proxy_preset.setEnabled(self._proxy_gen.isChecked())
        if self._proxy_use.isChecked():
            self.proxy_source_changed.emit("use_proxy")
        elif self._proxy_gen.isChecked():
            self.proxy_source_changed.emit("generate")
        else:
            self.proxy_source_changed.emit("full")

    # ── Public API ─────────────────────────────────────────────────────────

    def update_clip_info(self, clip: ClipInfo) -> None:
        p = Path(clip.file_path)
        fps_str = f"{clip.fps:.3f}".rstrip("0").rstrip(".")
        dur_str = f"{clip.duration_seconds:.1f}s ({clip.duration_frames} fr)"
        self._info_fields["file"].setText(p.name)
        self._info_fields["path"].setText(str(p.parent))
        self._info_fields["res"].setText(clip.resolution_label)
        self._info_fields["duration"].setText(dur_str)
        self._info_fields["fps"].setText(fps_str)
        self._info_fields["in_pt"].setText(str(clip.in_point_frame))
        self._info_fields["out_pt"].setText(str(clip.out_point_frame))

    def update_proxy_label(self, label: str) -> None:
        self._proxy_label.setText(label)

    def update_estimates(self, model: str, proxy: str, time: str) -> None:
        self._info_fields["est_model"].setText(model)
        self._info_fields["est_proxy"].setText(proxy)
        self._info_fields["est_time"].setText(time)

    @property
    def prompt_mode(self) -> str:
        if self._mode_points.isChecked(): return "points"
        if self._mode_box.isChecked():    return "box"
        if self._mode_mask.isChecked():   return "mask"
        return "text"


# ── Toolbar ─────────────────────────────────────────────────────────────────

class ToolBar(QWidget):
    tool_selected = pyqtSignal(str)  # 'pan' | 'point' | 'box'
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    fullscreen_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolbar_bar")
        self.setFixedHeight(40)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        def _tb(icon: str, tip: str, name: str = "", checkable: bool = False) -> QPushButton:
            btn = QPushButton(icon)
            btn.setObjectName("toolbar_btn")
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            if name:
                btn.setProperty("tool_name", name)
                btn.clicked.connect(lambda _, n=name: self.tool_selected.emit(n))
            return btn

        self._btn_pan   = _tb("✋", "Pan (Space)", "pan",   checkable=True)
        self._btn_point = _tb("✛", "Point mode (P)", "point", checkable=True)
        self._btn_box   = _tb("⬚", "Box mode (B)", "box",  checkable=True)
        self._btn_point.setChecked(True)

        btn_swap   = _tb("⇄", "Swap positive/negative")
        btn_undo   = _tb("↩", "Undo (Ctrl+Z)")
        btn_redo   = _tb("↪", "Redo (Ctrl+Y)")
        btn_delete = _tb("✕", "Delete selection")

        btn_undo.clicked.connect(self.undo_requested)
        btn_redo.clicked.connect(self.redo_requested)
        btn_delete.clicked.connect(self.delete_requested)

        for btn in (self._btn_pan, self._btn_point, self._btn_box):
            layout.addWidget(btn)
        layout.addWidget(_separator_v())
        for btn in (btn_swap, btn_undo, btn_redo, btn_delete):
            layout.addWidget(btn)

        layout.addStretch()

        # Zoom indicator
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #909090; font-size: 11px; min-width: 40px;")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.zoom_label)
        layout.addSpacing(4)

        btn_fs = _tb("⛶", "Fullscreen")
        btn_fs.clicked.connect(self.fullscreen_requested)
        layout.addWidget(btn_fs)

    def set_zoom(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")

    def set_active_tool(self, tool: str) -> None:
        for btn in (self._btn_pan, self._btn_point, self._btn_box):
            btn.setChecked(btn.property("tool_name") == tool)


def _separator_v() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setStyleSheet("color: #3A3A3A;")
    line.setFixedWidth(1)
    return line


# ── Canvas placeholder ──────────────────────────────────────────────────────

class CanvasPlaceholder(QWidget):
    """Placeholder shown before canvas_widget.py is wired in."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvas_container")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("Canvas — select a clip to begin")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #555555; font-size: 14px;")
        layout.addWidget(lbl)


# ── Scrubber + transport ────────────────────────────────────────────────────

class TransportBar(QWidget):
    frame_changed = pyqtSignal(int)
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._total_frames = 100
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scrubber
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setObjectName("scrubber")
        self.scrubber.setFixedHeight(8)
        self.scrubber.setRange(0, 100)
        self.scrubber.valueChanged.connect(self.frame_changed)
        layout.addWidget(self.scrubber)

        # Transport controls
        transport = QWidget()
        transport.setFixedHeight(36)
        t_layout = QHBoxLayout(transport)
        t_layout.setContentsMargins(8, 0, 8, 0)
        t_layout.setSpacing(4)

        def _tb(icon: str, tip: str) -> QPushButton:
            btn = QPushButton(icon)
            btn.setObjectName("toolbar_btn")
            btn.setToolTip(tip)
            btn.setFixedSize(QSize(28, 28))
            return btn

        self._btn_first  = _tb("|◀", "First frame")
        self._btn_prev   = _tb("◀",  "Previous frame")
        self._btn_play   = _tb("▶",  "Play / Pause (Space)")
        self._btn_play.setCheckable(True)
        self._btn_next   = _tb("▶",  "Next frame")
        self._btn_last   = _tb("▶|", "Last frame")

        self._btn_first.clicked.connect(lambda: self.frame_changed.emit(0))
        self._btn_prev.clicked.connect(self._step_back)
        self._btn_next.clicked.connect(self._step_fwd)
        self._btn_last.clicked.connect(lambda: self.frame_changed.emit(self._total_frames - 1))
        self._btn_play.toggled.connect(self.play_toggled)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setFixedWidth(64)
        self.speed_combo.currentTextChanged.connect(
            lambda t: self.speed_changed.emit(float(t.replace("x", "")))
        )

        t_layout.addStretch()
        for btn in (self._btn_first, self._btn_prev, self._btn_play, self._btn_next, self._btn_last):
            t_layout.addWidget(btn)
        t_layout.addSpacing(8)
        t_layout.addWidget(self.speed_combo)
        t_layout.addStretch()

        layout.addWidget(transport)

    def set_total_frames(self, n: int) -> None:
        self._total_frames = max(1, n)
        self.scrubber.setRange(0, self._total_frames - 1)

    def set_frame(self, idx: int) -> None:
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(idx)
        self.scrubber.blockSignals(False)

    def _step_back(self) -> None:
        self.frame_changed.emit(max(0, self.scrubber.value() - 1))

    def _step_fwd(self) -> None:
        self.frame_changed.emit(min(self._total_frames - 1, self.scrubber.value() + 1))


# ── Thumbnail strip ─────────────────────────────────────────────────────────

class ThumbnailStrip(QWidget):
    frame_selected = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("thumbnail_strip_container")
        self.setFixedHeight(88)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 0)
        outer.setSpacing(2)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedHeight(68)

        self._strip_content = QWidget()
        self._strip_layout = QHBoxLayout(self._strip_content)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(3)
        self._strip_layout.addStretch()

        scroll.setWidget(self._strip_content)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(12)
        for label, obj_name, color in [
            ("■ High Confidence", "conf_high", "#5AB85A"),
            ("■ Medium",         "conf_med",  "#E8A030"),
            ("■ Low",            "conf_low",  "#E85050"),
            ("□ No Mask",        "conf_none", "#444444"),
        ]:
            lbl = QLabel(label)
            lbl.setObjectName(obj_name)
            lbl.setStyleSheet(f"color: {color}; font-size: 10px;")
            legend.addWidget(lbl)
        legend.addStretch()
        outer.addLayout(legend)


# ── Context bar ─────────────────────────────────────────────────────────────

class ContextBar(QWidget):
    undo_requested = pyqtSignal()
    clear_all_requested = pyqtSignal()

    _MODE_HINTS = {
        "points": ("POINTS MODE:", "Left-click positive · Right-click negative"),
        "box":    ("BOX MODE:",    "Click and drag to draw a bounding box"),
        "mask":   ("MASK MODE:",   "Load a PNG mask file as prompt input"),
        "text":   ("TEXT MODE:",   "Enter a description and click Detect"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("context_bar")
        self.setFixedHeight(26)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        self._mode_lbl = QLabel("POINTS MODE:")
        self._mode_lbl.setObjectName("context_mode_label")

        self._hint_lbl = QLabel("Left-click positive · Right-click negative")
        self._hint_lbl.setObjectName("context_hint_label")

        layout.addWidget(self._mode_lbl)
        layout.addWidget(self._hint_lbl)
        layout.addStretch()

        undo_lbl = QLabel("Ctrl+Z")
        undo_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        undo_btn = QPushButton("Undo")
        undo_btn.setObjectName("refresh_btn")
        undo_btn.setFixedHeight(20)
        undo_btn.clicked.connect(self.undo_requested)

        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("refresh_btn")
        clear_btn.setFixedHeight(20)
        clear_btn.clicked.connect(self.clear_all_requested)

        layout.addWidget(undo_lbl)
        layout.addWidget(undo_btn)
        layout.addWidget(clear_btn)

    def set_mode(self, mode: str) -> None:
        title, hint = self._MODE_HINTS.get(mode, ("", ""))
        self._mode_lbl.setText(title)
        self._hint_lbl.setText(hint)


# ── Action buttons bar ──────────────────────────────────────────────────────

class ActionBar(QWidget):
    run_requested     = pyqtSignal()
    preview_requested = pyqtSignal()
    accept_requested  = pyqtSignal()
    modify_requested  = pyqtSignal()
    cancel_requested  = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("action_bar")
        self.setFixedHeight(52)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        specs = [
            ("btn_run",     "Run SAM3 ▶",  "Track Objects",       self.run_requested),
            ("btn_preview", "Preview ▶",   "Review Results",      self.preview_requested),
            ("btn_accept",  "Accept",       "Import to Resolve",   self.accept_requested),
            ("btn_modify",  "Modify",       "Edit & Re-propagate", self.modify_requested),
            ("btn_cancel",  "Cancel",       "Discard All",         self.cancel_requested),
        ]

        self._buttons: dict[str, QPushButton] = {}

        for obj_name, main_label, sub_label, signal in specs:
            container = QWidget()
            container.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(1)

            btn = QPushButton(main_label)
            btn.setObjectName(obj_name)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.clicked.connect(signal)
            self._buttons[obj_name] = btn

            sub = QLabel(sub_label)
            sub.setObjectName("btn_subtitle")
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

            vbox.addWidget(btn)
            vbox.addWidget(sub)
            layout.addWidget(container)

        # Initial disabled state
        self.set_propagation_complete(False)
        self.set_has_prompts(False)

    def set_has_prompts(self, has: bool) -> None:
        self._buttons["btn_run"].setEnabled(has)

    def set_propagation_complete(self, done: bool) -> None:
        self._buttons["btn_preview"].setEnabled(done)
        self._buttons["btn_accept"].setEnabled(done)
        self._buttons["btn_modify"].setEnabled(done)


# ── Right panel ─────────────────────────────────────────────────────────────

class RightPanel(QWidget):
    add_object_requested = pyqtSignal()
    text_detect_requested = pyqtSignal(str)
    text_result_accepted = pyqtSignal()
    text_result_rejected = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("right_panel")
        self.setFixedWidth(240)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        # ── Objects header ────────────────────────────────────
        obj_header = QHBoxLayout()
        self._objects_title = QLabel("OBJECTS (0 / 8)")
        self._objects_title.setObjectName("section_label")
        add_btn = QPushButton("+ Add Object")
        add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self.add_object_requested)
        self._add_obj_btn = add_btn
        obj_header.addWidget(self._objects_title)
        obj_header.addStretch()
        obj_header.addWidget(add_btn)
        layout.addLayout(obj_header)

        # Placeholder — object rows injected by object_panel.py
        self.objects_container = QWidget()
        self.objects_layout = QVBoxLayout(self.objects_container)
        self.objects_layout.setContentsMargins(0, 0, 0, 0)
        self.objects_layout.setSpacing(2)
        layout.addWidget(self.objects_container)

        layout.addSpacing(8)

        # ── Preview Options ───────────────────────────────────
        layout.addWidget(_section_label("Preview Options"))

        self.display_mode = QComboBox()
        self.display_mode.addItems(["Overlay", "Matte", "Cutout", "Outline"])
        layout.addWidget(self.display_mode)

        self.cb_show_all  = QCheckBox("Show All Objects")
        self.cb_show_conf = QCheckBox("Show Confidence")
        self.cb_show_empty = QCheckBox("Show Empty Frames")
        self.cb_show_all.setChecked(True)

        for cb in (self.cb_show_all, self.cb_show_conf, self.cb_show_empty):
            layout.addWidget(cb)

        self.preview_speed = QComboBox()
        self.preview_speed.addItems(["0.25x", "0.5x", "1.0x"])
        self.preview_speed.setCurrentText("1.0x")
        layout.addWidget(self.preview_speed)

        layout.addSpacing(8)

        # ── Text Prompt ───────────────────────────────────────
        layout.addWidget(_section_label("Text Prompt (SAM3)"))

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Describe the object…")
        layout.addWidget(self.text_input)

        detect_btn = QPushButton("Detect")
        detect_btn.clicked.connect(
            lambda: self.text_detect_requested.emit(self.text_input.text())
        )
        layout.addWidget(detect_btn)

        # Detection result card (hidden until detection runs)
        self._detection_card = QWidget()
        self._detection_card.setObjectName("detection_card")
        self._detection_card.hide()
        card_layout = QVBoxLayout(self._detection_card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)
        self._detection_result_lbl = QLabel()
        self._detection_result_lbl.setObjectName("detection_label")
        self._detection_result_lbl.setWordWrap(True)
        card_layout.addWidget(self._detection_result_lbl)

        card_btns = QHBoxLayout()
        accept_d = QPushButton("Accept")
        reject_d = QPushButton("Reject")
        accept_d.setFixedHeight(22)
        reject_d.setFixedHeight(22)
        accept_d.clicked.connect(self.text_result_accepted)
        reject_d.clicked.connect(self.text_result_rejected)
        card_btns.addWidget(accept_d)
        card_btns.addWidget(reject_d)
        card_layout.addLayout(card_btns)

        layout.addWidget(self._detection_card)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def update_object_count(self, n: int) -> None:
        self._objects_title.setText(f"OBJECTS ({n} / {MAX_OBJECTS})")
        self._add_obj_btn.setEnabled(n < MAX_OBJECTS)

    def show_detection_result(self, label: str, confidence: float) -> None:
        conf_pct = int(confidence * 100)
        text = f"Detected: {label} ({conf_pct}% conf)"
        self._detection_result_lbl.setText(text)
        if confidence < 0.5:
            self._detection_result_lbl.setObjectName("detection_conf_warn")
        else:
            self._detection_result_lbl.setObjectName("detection_label")
        self._detection_card.show()

    def hide_detection_result(self) -> None:
        self._detection_card.hide()


# ── Bottom panel ────────────────────────────────────────────────────────────

class BottomPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("bottom_panel")
        self.setFixedHeight(160)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tabs: Log + Progress
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Log tab
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("log_view")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.tabs.addTab(self.log_view, "LOG")

        # Progress tab
        prog_widget = QWidget()
        prog_layout = QVBoxLayout(prog_widget)
        prog_layout.setContentsMargins(8, 8, 8, 8)
        prog_layout.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.progress_detail = QLabel("Ready")
        self.progress_detail.setStyleSheet("color: #909090; font-size: 11px;")

        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.progress_detail)
        prog_layout.addStretch()
        self.tabs.addTab(prog_widget, "PROGRESS")

        layout.addWidget(self.tabs, stretch=1)

        # Quick Stats (always visible, right side)
        stats = QWidget()
        stats.setObjectName("quick_stats_panel")
        stats.setFixedWidth(180)
        stats_layout = QGridLayout(stats)
        stats_layout.setContentsMargins(10, 8, 10, 8)
        stats_layout.setHorizontalSpacing(8)
        stats_layout.setVerticalSpacing(4)

        self._stat_widgets: dict[str, QLabel] = {}
        for row, (key, attr, default) in enumerate([
            ("Frames",    "frames",    "—"),
            ("Processed", "processed", "— (—%)"),
            ("Speed",     "speed",     "— fps"),
            ("ETA",       "eta",       "--:--"),
        ]):
            k = QLabel(key)
            k.setObjectName("stat_key")
            v = QLabel(default)
            v.setObjectName("stat_value")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            stats_layout.addWidget(k, row, 0)
            stats_layout.addWidget(v, row, 1)
            self._stat_widgets[attr] = v

        layout.addWidget(stats)

    def append_log(self, level: str, message: str) -> None:
        """Append a timestamped log line with colour coding."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO":  "#909090",
            "OK":    "#5AB85A",
            "WARN":  "#E8A030",
            "ERROR": "#E85050",
        }
        color = color_map.get(level.upper(), "#909090")
        html = (
            f'<span style="color:#555555">{ts}</span>&nbsp;'
            f'<span style="color:{color}">[{level}]</span>&nbsp;'
            f'<span style="color:#E8E8E8">{message}</span>'
        )
        self.log_view.appendHtml(html)

    def update_progress(
        self, current: int, total: int, fps: float, eta_s: float
    ) -> None:
        pct = int(current / total * 100) if total else 0
        self.progress_bar.setValue(pct)
        eta_str = f"{int(eta_s // 60):02d}:{int(eta_s % 60):02d}"
        self.progress_detail.setText(
            f"{current} / {total} frames  ·  {fps:.1f} fps  ·  ETA {eta_str}"
        )
        self._stat_widgets["processed"].setText(f"{current} ({pct}%)")
        self._stat_widgets["speed"].setText(f"{fps:.1f} fps")
        self._stat_widgets["eta"].setText(eta_str)

    def set_total_frames(self, n: int) -> None:
        self._stat_widgets["frames"].setText(str(n))


# ── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Top-level application window.

    Assembles all panels and wires their signals together.
    Canvas, object rows, and settings slide-in are injected in later build steps.
    """

    def __init__(
        self,
        clip: Optional[ClipInfo] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAM3 Mask Tracker")
        self.setMinimumSize(1200, 700)

        # Load QSS
        app = QApplication.instance()
        if app:
            app.setStyleSheet(_load_stylesheet())

        self._clip = clip
        self._current_frame = 0
        self._build_ui()
        self._wire_signals()

        if clip:
            self._apply_clip(clip)
        else:
            self.header.update_gpu_status("No clip selected", ready=False)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────
        self.header = HeaderBar()
        root.addWidget(self.header)

        # ── Body (left + canvas column + right) ─────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.left_panel = LeftPanel()
        body.addWidget(self.left_panel)

        # Canvas column: toolbar + canvas + scrubber + thumbnail + context + actions
        canvas_col = QVBoxLayout()
        canvas_col.setContentsMargins(0, 0, 0, 0)
        canvas_col.setSpacing(0)

        self.toolbar = ToolBar()
        canvas_col.addWidget(self.toolbar)

        # Interactive canvas (replaces placeholder)
        from sam3_resolve.ui.canvas_widget import CanvasWidget
        from sam3_resolve.core.sam3_runner import create_runner
        self._runner = create_runner(force_mock=True)
        self._runner.load_model()
        self.canvas = CanvasWidget(runner=self._runner)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Add default object 1 so the canvas is ready immediately
        self.canvas.add_object(1)
        canvas_col.addWidget(self.canvas, stretch=1)

        self.transport = TransportBar()
        canvas_col.addWidget(self.transport)

        self.thumbnails = ThumbnailStrip()
        canvas_col.addWidget(self.thumbnails)

        self.context_bar = ContextBar()
        canvas_col.addWidget(self.context_bar)

        self.action_bar = ActionBar()
        canvas_col.addWidget(self.action_bar)

        body.addLayout(canvas_col, stretch=1)

        self.right_panel = RightPanel()
        body.addWidget(self.right_panel)

        root.addLayout(body, stretch=1)

        # ── Bottom ──────────────────────────────────────
        self.bottom_panel = BottomPanel()
        root.addWidget(self.bottom_panel)

    def _wire_signals(self) -> None:
        self.header.settings_requested.connect(self._toggle_settings)
        self.left_panel.prompt_mode_changed.connect(self._on_prompt_mode_changed)
        self.context_bar.clear_all_requested.connect(self._on_clear_all)
        self.context_bar.undo_requested.connect(self._on_undo)
        self.action_bar.run_requested.connect(self._on_run)
        self.action_bar.preview_requested.connect(self._on_preview)
        self.action_bar.accept_requested.connect(self._on_accept)
        self.action_bar.modify_requested.connect(self._on_modify)
        self.action_bar.cancel_requested.connect(self._on_cancel)
        self.transport.frame_changed.connect(self._on_frame_changed)
        self.right_panel.add_object_requested.connect(self._on_add_object)
        # Canvas ↔ UI
        self.canvas.prompts_changed.connect(self._on_prompts_changed)
        self.canvas.inference_requested.connect(self._on_inference_requested)
        self.toolbar.undo_requested.connect(self.canvas.undo_last_prompt)

    # ── Clip ───────────────────────────────────────────────────────────────

    def _apply_clip(self, clip: ClipInfo) -> None:
        self._clip = clip
        self.header.update_clip_info(clip)
        self.left_panel.update_clip_info(clip)
        self.transport.set_total_frames(clip.duration_frames)
        self.bottom_panel.set_total_frames(clip.duration_frames)
        self.bottom_panel.append_log("OK", f"Clip loaded: {clip.name}")

    # ── Signal handlers ────────────────────────────────────────────────────

    def _on_prompt_mode_changed(self, mode: str) -> None:
        self.context_bar.set_mode(mode)
        self.toolbar.set_active_tool(mode if mode in ("pan", "point", "box") else "point")
        self.bottom_panel.append_log("INFO", f"Prompt mode → {mode}")

    def _on_frame_changed(self, idx: int) -> None:
        self._current_frame = idx
        self.transport.set_frame(idx)

    def _on_clear_all(self) -> None:
        self.bottom_panel.append_log("INFO", "All prompts cleared")

    def _on_undo(self) -> None:
        self.bottom_panel.append_log("INFO", "Undo last prompt")

    def _on_run(self) -> None:
        self.bottom_panel.append_log("INFO", "Starting SAM3 propagation…")
        self.action_bar.set_propagation_complete(False)

    def _on_preview(self) -> None:
        self.bottom_panel.append_log("INFO", "Opening preview player")

    def _on_accept(self) -> None:
        self.bottom_panel.append_log("INFO", "Importing masks to Resolve Fusion…")

    def _on_modify(self) -> None:
        self.bottom_panel.append_log("INFO", "Returning to edit mode")
        self.action_bar.set_propagation_complete(False)

    def _on_cancel(self) -> None:
        self.bottom_panel.append_log("WARN", "Propagation discarded")
        self.action_bar.set_propagation_complete(False)
        self.action_bar.set_has_prompts(False)

    def _on_add_object(self) -> None:
        n = len(self.canvas._objects) + 1
        if n > 8:
            return
        self.canvas.add_object(n)
        self.canvas.set_active_object(n)
        self.right_panel.update_object_count(n)
        self.bottom_panel.append_log("INFO", f"New tracking object {n} added")

    def _on_prompts_changed(self, obj_id: int) -> None:
        has = any(
            obj.prompts.has_prompts()
            for obj in self.canvas._objects.values()
        )
        self.action_bar.set_has_prompts(has)
        self.toolbar.set_zoom(self.canvas.zoom_percent)

    def _on_inference_requested(self) -> None:
        """Launch live single-frame inference in a worker thread."""
        from sam3_resolve.ui.workers import LiveInferenceWorker
        if self.canvas._frame is None:
            return
        self.canvas.set_inference_running(True)
        worker = LiveInferenceWorker(runner=self._runner, parent=self)
        worker.set_task(self.canvas._frame_idx, self.canvas._frame)
        worker.mask_ready.connect(self._on_live_mask_ready)
        worker.finished.connect(lambda: self.canvas.set_inference_running(False))
        worker.start()
        self._live_worker = worker   # keep reference

    def _on_live_mask_ready(self, masks: dict) -> None:
        self.canvas.set_all_masks(masks)
        self.canvas.set_inference_running(False)
        self.toolbar.set_zoom(self.canvas.zoom_percent)

    def _toggle_settings(self) -> None:
        self.bottom_panel.append_log("INFO", "Settings panel toggled")

    # ── GPU status (called after gpu_utils.detect_gpu) ────────────────────

    def set_gpu_info_label(self, label: str, ready: bool = True) -> None:
        self.header.update_gpu_status(label, ready)
