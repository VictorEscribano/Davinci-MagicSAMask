"""
Post-propagation preview player.

Displays the result of a full SAM3 propagation with four render modes:
  Overlay  — semi-transparent mask fill over original frame
  Matte    — white mask on black background (per-object or composite)
  Cutout   — subject pixels on transparent/checker background
  Outline  — original frame with coloured contour lines only

Controls:
  Play / Pause / Scrub / Speed (0.25× – 4×)
  Object visibility toggles
  "Modify" button — emits modify_requested so caller can return to canvas
  "Export" button — emits export_requested so caller can open MaskExporter

The player never touches disk; it receives PropagationResult in-memory and
reads original frames on demand via MediaHandler.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sam3_resolve.core.media_handler import MediaHandler
from sam3_resolve.core.sam3_runner import PropagationResult
from sam3_resolve.ui.canvas_widget import OBJECT_COLORS

logger = logging.getLogger(__name__)

DISPLAY_MODES = ["Overlay", "Matte", "Cutout", "Outline"]
SPEED_OPTIONS  = [0.25, 0.5, 1.0, 2.0, 4.0]
OVERLAY_ALPHA  = 160   # 0-255 fill alpha for Overlay mode
OUTLINE_WIDTH  = 2     # px contour line width


# ── Render helpers (pure functions — easy to unit-test) ────────────────────

def render_overlay(
    bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    colors: dict[int, QColor],
    alpha: int = OVERLAY_ALPHA,
) -> np.ndarray:
    """Return BGR frame with coloured mask fills blended on top."""
    out = bgr.copy()
    for obj_id, mask in masks.items():
        if mask is None or mask.max() == 0:
            continue
        color = colors.get(obj_id, QColor("#4A9EFF"))
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[mask > 127] = [color.blue(), color.green(), color.red()]
        a = alpha / 255.0
        out = np.where(
            (mask > 127)[:, :, None],
            (out * (1 - a) + overlay * a).clip(0, 255).astype(np.uint8),
            out,
        )
    return out


def render_matte(
    bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    colors: dict[int, QColor],
    composite: bool = True,
) -> np.ndarray:
    """Return white-on-black matte.  composite=True merges all objects."""
    h, w = bgr.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id, mask in masks.items():
        if mask is None:
            continue
        c = colors.get(obj_id, QColor("white"))
        out[mask > 127] = [c.blue(), c.green(), c.red()] if not composite else [255, 255, 255]
    return out


def render_cutout(
    bgr: np.ndarray,
    masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Return BGRA with alpha set from combined mask (checker fill for zero areas)."""
    h, w = bgr.shape[:2]
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks.values():
        if mask is not None:
            combined = np.maximum(combined, mask)
    bgra[:, :, 3] = combined
    # Dark checker background where alpha=0
    checker = _checker_background(h, w)
    alpha_f = combined[:, :, None] / 255.0
    merged = (bgr * alpha_f + checker * (1 - alpha_f)).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(merged, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = 255
    return result


def render_outline(
    bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    colors: dict[int, QColor],
    line_width: int = OUTLINE_WIDTH,
) -> np.ndarray:
    """Return original frame with coloured contour outlines drawn over it."""
    out = bgr.copy()
    for obj_id, mask in masks.items():
        if mask is None or mask.max() == 0:
            continue
        color = colors.get(obj_id, QColor("#4A9EFF"))
        bgr_color = (color.blue(), color.green(), color.red())
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, bgr_color, line_width)
    return out


def _checker_background(h: int, w: int, cell: int = 16) -> np.ndarray:
    """Generate a grey checkerboard array (H, W, 3)."""
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    for row in range(0, h, cell):
        for col in range(0, w, cell):
            if (row // cell + col // cell) % 2 == 0:
                bg[row:row + cell, col:col + cell] = 60
    return bg


def bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ── Canvas area that paints the rendered frame ─────────────────────────────

class _PreviewCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 225)

    def set_pixmap(self, px: QPixmap) -> None:
        self._pixmap = px
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1C1C1C"))
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


# ── Preview Player Widget ──────────────────────────────────────────────────

class PreviewPlayer(QWidget):
    """
    Full post-propagation review widget.

    Signals:
        modify_requested():  user wants to go back to canvas prompting
        export_requested():  user wants to open MaskExporter
    """

    modify_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(
        self,
        result: PropagationResult,
        media: MediaHandler,
        object_colors: Optional[dict[int, QColor]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._media  = media
        self._colors: dict[int, QColor] = object_colors or {}
        self._frame_indices = sorted(result.masks.keys())
        self._current_idx   = 0          # index into _frame_indices
        self._playing       = False
        self._speed         = 1.0
        self._mode          = "Overlay"
        self._visible: dict[int, bool] = {oid: True for oid in self._all_object_ids()}

        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build()
        self._show_frame(0)

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Canvas
        self._canvas = _PreviewCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # Controls row 1: mode + object visibility
        ctrl1 = QHBoxLayout()
        ctrl1.setContentsMargins(8, 4, 8, 0)
        ctrl1.setSpacing(8)

        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet("color: #909090; font-size: 11px;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(DISPLAY_MODES)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        ctrl1.addWidget(mode_lbl)
        ctrl1.addWidget(self._mode_combo)
        ctrl1.addStretch()

        # Per-object visibility checkboxes
        self._vis_checks: dict[int, QCheckBox] = {}
        for obj_id in self._all_object_ids():
            color = self._colors.get(obj_id, QColor("#4A9EFF"))
            cb = QCheckBox(f"Obj {obj_id}")
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color.name()}; font-size: 11px;")
            cb.toggled.connect(lambda checked, oid=obj_id: self._on_visibility(oid, checked))
            self._vis_checks[obj_id] = cb
            ctrl1.addWidget(cb)

        layout.addLayout(ctrl1)

        # Scrubber
        scrub_row = QHBoxLayout()
        scrub_row.setContentsMargins(8, 4, 8, 0)
        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setStyleSheet("color: #909090; font-size: 10px;")
        self._frame_lbl.setFixedWidth(60)
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, max(0, len(self._frame_indices) - 1))
        self._scrubber.valueChanged.connect(self._on_scrub)
        scrub_row.addWidget(self._frame_lbl)
        scrub_row.addWidget(self._scrubber)
        layout.addLayout(scrub_row)

        # Transport row
        transport = QHBoxLayout()
        transport.setContentsMargins(8, 4, 8, 4)
        transport.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._toggle_play)

        btn_prev = QPushButton("◀")
        btn_prev.setFixedWidth(28)
        btn_prev.clicked.connect(lambda: self._step(-1))

        btn_next = QPushButton("▶")
        btn_next.setFixedWidth(28)
        btn_next.clicked.connect(lambda: self._step(1))

        speed_lbl = QLabel("Speed:")
        speed_lbl.setStyleSheet("color: #909090; font-size: 11px;")
        self._speed_combo = QComboBox()
        for s in SPEED_OPTIONS:
            self._speed_combo.addItem(f"{s}×", s)
        self._speed_combo.setCurrentIndex(2)  # 1×
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        transport.addWidget(btn_prev)
        transport.addWidget(self._play_btn)
        transport.addWidget(btn_next)
        transport.addStretch()
        transport.addWidget(speed_lbl)
        transport.addWidget(self._speed_combo)

        # Action buttons
        btn_modify = QPushButton("✎  Modify")
        btn_modify.setObjectName("btn_run")
        btn_modify.clicked.connect(self.modify_requested)

        btn_export = QPushButton("⬇  Export Masks")
        btn_export.setObjectName("btn_accept")
        btn_export.clicked.connect(self.export_requested)

        transport.addWidget(btn_modify)
        transport.addWidget(btn_export)
        layout.addLayout(transport)

    # ── Playback ──────────────────────────────────────────────────────────

    def _toggle_play(self, checked: bool) -> None:
        self._playing = checked
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")
        if checked:
            self._schedule_next()

    def _schedule_next(self) -> None:
        if not self._playing:
            return
        fps = self._media.clip.fps if self._media and self._media.clip else 24.0
        interval_ms = max(1, int(1000 / (fps * self._speed)))
        self._play_timer.start(interval_ms)

    def _advance_frame(self) -> None:
        if not self._playing:
            return
        next_idx = self._current_idx + 1
        if next_idx >= len(self._frame_indices):
            next_idx = 0  # loop
        self._show_frame(next_idx)
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(next_idx)
        self._scrubber.blockSignals(False)
        self._schedule_next()

    def _step(self, delta: int) -> None:
        target = max(0, min(len(self._frame_indices) - 1, self._current_idx + delta))
        self._show_frame(target)
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(target)
        self._scrubber.blockSignals(False)

    def _on_scrub(self, pos: int) -> None:
        self._show_frame(pos)

    def _on_mode_changed(self, mode: str) -> None:
        self._mode = mode
        self._show_frame(self._current_idx)

    def _on_visibility(self, obj_id: int, visible: bool) -> None:
        self._visible[obj_id] = visible
        self._show_frame(self._current_idx)

    def _on_speed_changed(self, _idx: int) -> None:
        self._speed = self._speed_combo.currentData()

    # ── Rendering ─────────────────────────────────────────────────────────

    def _show_frame(self, list_idx: int) -> None:
        if not self._frame_indices:
            return
        list_idx = max(0, min(list_idx, len(self._frame_indices) - 1))
        self._current_idx = list_idx
        frame_idx = self._frame_indices[list_idx]

        # Load source frame
        try:
            bgr = self._media.read_frame(frame_idx, use_cache=True, proxy=False)
        except Exception:
            bgr = np.zeros((360, 640, 3), dtype=np.uint8)

        # Gather visible masks for this frame
        frame_masks = self._result.masks.get(frame_idx, {})
        visible_masks: dict[int, np.ndarray] = {
            oid: m for oid, m in frame_masks.items()
            if self._visible.get(oid, True)
        }

        rendered = self._render(bgr, visible_masks)
        px = bgr_to_pixmap(rendered)
        self._canvas.set_pixmap(px)

        # Update frame label
        total = len(self._frame_indices)
        self._frame_lbl.setText(f"{list_idx + 1} / {total}")

    def _render(self, bgr: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
        dispatch = {
            "Overlay": lambda: render_overlay(bgr, masks, self._colors),
            "Matte":   lambda: render_matte(bgr, masks, self._colors),
            "Cutout":  lambda: render_cutout(bgr, masks)[:, :, :3],
            "Outline": lambda: render_outline(bgr, masks, self._colors),
        }
        fn = dispatch.get(self._mode, dispatch["Overlay"])
        return fn()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _all_object_ids(self) -> list[int]:
        ids: set[int] = set()
        for frame_masks in self._result.masks.values():
            ids.update(frame_masks.keys())
        return sorted(ids)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_result(self, result: PropagationResult) -> None:
        """Replace propagation result (e.g. after re-run)."""
        self._result = result
        self._frame_indices = sorted(result.masks.keys())
        self._scrubber.setRange(0, max(0, len(self._frame_indices) - 1))
        self._show_frame(0)

    def current_frame_index(self) -> int:
        """Return absolute frame index currently displayed."""
        if not self._frame_indices:
            return 0
        return self._frame_indices[self._current_idx]

    def stop(self) -> None:
        """Stop playback (call before closing widget)."""
        self._playing = False
        self._play_timer.stop()
