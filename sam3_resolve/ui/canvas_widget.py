"""
Interactive canvas widget.

Responsibilities:
  - Display video frame letterboxed inside the canvas area
  - Zoom (scroll wheel) and pan (middle-click drag)
  - Accept prompt inputs: point clicks, box drag, undo
  - Render mask overlays (per-object semi-transparent fill + white contour)
  - Show positive/negative prompt points with order numbers
  - Show bounding box (dashed while drawing, solid after release)
  - Frame counter overlay (bottom-left)
  - Debounced live-inference trigger (80 ms)
  - Spinning inference indicator while waiting for mask
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from sam3_resolve.constants import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    LIVE_INFERENCE_DEBOUNCE_MS,
    MAX_OBJECTS,
)
from sam3_resolve.core.sam3_runner import (
    ObjectPrompts,
    PromptBox,
    PromptPoint,
    SAM3RunnerBase,
)

logger = logging.getLogger(__name__)

# ── Object colour palette ──────────────────────────────────────────────────

OBJECT_COLORS = [
    QColor(0x4A, 0x9E, 0xFF),   # blue
    QColor(0xFF, 0x80, 0x00),   # orange
    QColor(0x80, 0xFF, 0x00),   # lime
    QColor(0xFF, 0x00, 0x80),   # pink
    QColor(0x00, 0xFF, 0xFF),   # cyan
    QColor(0xFF, 0xFF, 0x00),   # yellow
    QColor(0xFF, 0x00, 0x00),   # red
    QColor(0x80, 0x00, 0xFF),   # purple
]

POINT_RADIUS = 6          # half-diameter for the click circles
POINT_FONT_SIZE = 8


# ── Internal state types ───────────────────────────────────────────────────

@dataclass
class ObjectState:
    """UI state for one tracked object."""
    object_id: int
    color: QColor
    opacity: int = 70          # 0-100
    visible: bool = True
    prompts: ObjectPrompts = field(default_factory=lambda: ObjectPrompts(object_id=0))
    mask: Optional[np.ndarray] = None    # last predicted mask (H, W, uint8)

    def __post_init__(self) -> None:
        self.prompts = ObjectPrompts(object_id=self.object_id)


# ── Canvas widget ──────────────────────────────────────────────────────────

class CanvasWidget(QWidget):
    """
    Interactive annotation and preview canvas.

    Signals:
        prompts_changed(int):          Active object ID whose prompts changed.
        frame_requested(int):          User scrubbed to a frame index.
        inference_requested():         Debounced trigger for live inference.
    """

    prompts_changed = pyqtSignal(int)
    frame_requested = pyqtSignal(int)
    inference_requested = pyqtSignal()

    def __init__(
        self,
        runner: Optional[SAM3RunnerBase] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("canvas_container")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._runner = runner

        # ── Frame data ─────────────────────────────────────────────────────
        self._frame: Optional[np.ndarray] = None       # BGR numpy from OpenCV
        self._frame_pixmap: Optional[QPixmap] = None   # rendered QPixmap
        self._frame_idx: int = 0
        self._total_frames: int = 1

        # ── Object / prompt state ──────────────────────────────────────────
        self._objects: dict[int, ObjectState] = {}
        self._active_object_id: int = 1

        # ── Mode ───────────────────────────────────────────────────────────
        self._mode: str = "points"   # 'points' | 'box' | 'mask' | 'text' | 'pan'

        # ── Zoom / pan ─────────────────────────────────────────────────────
        self._zoom: float = 1.0
        self._pan_offset: QPointF = QPointF(0, 0)
        self._pan_last: Optional[QPoint] = None

        # ── Box drawing ────────────────────────────────────────────────────
        self._box_start: Optional[QPointF] = None     # in video coords
        self._box_end:   Optional[QPointF] = None
        self._box_drawing: bool = False

        # ── Live inference ─────────────────────────────────────────────────
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(LIVE_INFERENCE_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self.inference_requested)
        self._inference_running: bool = False

        # Spinning indicator timer
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(120)
        self._spin_timer.timeout.connect(self._advance_spinner)
        self._spin_frame: int = 0

        # ── Undo history ───────────────────────────────────────────────────
        # Each entry: ('point', obj_id, PromptPoint) or ('box', obj_id)
        self._undo_stack: list[tuple] = []

        self.setMinimumSize(500, 300)
        self._update_cursor()

    # ── Public API ─────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray, frame_idx: int, total: int) -> None:
        """
        Load a new video frame into the canvas.

        Args:
            frame:     BGR numpy array (H, W, 3).
            frame_idx: Absolute frame index.
            total:     Total frame count (for counter overlay).
        """
        self._frame = frame
        self._frame_idx = frame_idx
        self._total_frames = total
        self._frame_pixmap = self._bgr_to_pixmap(frame)
        self.update()

    def set_mode(self, mode: str) -> None:
        """Switch prompt mode ('points', 'box', 'mask', 'text', 'pan')."""
        self._mode = mode
        self._box_drawing = False
        self._box_start = None
        self._box_end = None
        self._update_cursor()
        self.update()

    def set_active_object(self, object_id: int) -> None:
        """Set which object receives new prompts."""
        if object_id not in self._objects:
            self._add_object(object_id)
        self._active_object_id = object_id
        self.update()

    def add_object(self, object_id: int, color: Optional[QColor] = None) -> None:
        """Register a new tracked object."""
        self._add_object(object_id, color)
        self.update()

    def remove_object(self, object_id: int) -> None:
        self._objects.pop(object_id, None)
        self.update()

    def set_object_mask(
        self, object_id: int, mask: np.ndarray
    ) -> None:
        """Update the displayed mask for one object."""
        if object_id in self._objects:
            self._objects[object_id].mask = mask
        self.update()

    def set_all_masks(self, masks: dict[int, np.ndarray]) -> None:
        """Update masks for all objects at once (called after propagation)."""
        for obj_id, mask in masks.items():
            if obj_id in self._objects:
                self._objects[obj_id].mask = mask
        self.update()

    def set_object_opacity(self, object_id: int, opacity: int) -> None:
        if object_id in self._objects:
            self._objects[object_id].opacity = opacity
            self.update()

    def set_object_visible(self, object_id: int, visible: bool) -> None:
        if object_id in self._objects:
            self._objects[object_id].visible = visible
            self.update()

    def set_inference_running(self, running: bool) -> None:
        self._inference_running = running
        if running:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.update()

    def undo_last_prompt(self) -> None:
        """Remove the most recent prompt element and re-trigger live inference."""
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        kind = action[0]
        obj_id = action[1]

        if kind == "point" and obj_id in self._objects:
            pts = self._objects[obj_id].prompts.points
            if pts:
                pts.pop()
            self._sync_runner_prompts(obj_id)

        elif kind == "box" and obj_id in self._objects:
            self._objects[obj_id].prompts.box = None
            self._box_start = None
            self._box_end = None
            self._sync_runner_prompts(obj_id)

        self._schedule_inference()
        self.prompts_changed.emit(obj_id)
        self.update()

    def clear_all_prompts(self) -> None:
        """Clear all prompts for all objects and masks."""
        for obj in self._objects.values():
            obj.prompts.points.clear()
            obj.prompts.box = None
            obj.mask = None
        self._undo_stack.clear()
        self._box_start = None
        self._box_end = None
        if self._runner:
            self._runner.clear_all_prompts()
        self.update()

    def get_all_prompts(self) -> dict[int, ObjectPrompts]:
        """Return current prompts for all objects."""
        return {oid: obj.prompts for oid, obj in self._objects.items()}

    def set_zoom(self, factor: float) -> None:
        self._zoom = max(0.1, min(factor, 8.0))
        self.update()

    @property
    def zoom_percent(self) -> int:
        return int(self._zoom * 100)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _add_object(
        self, object_id: int, color: Optional[QColor] = None
    ) -> None:
        if object_id in self._objects:
            return
        c = color or OBJECT_COLORS[(object_id - 1) % len(OBJECT_COLORS)]
        self._objects[object_id] = ObjectState(object_id=object_id, color=c)

    def _active_obj(self) -> Optional[ObjectState]:
        if self._active_object_id not in self._objects:
            self._add_object(self._active_object_id)
        return self._objects[self._active_object_id]

    def _sync_runner_prompts(self, object_id: int) -> None:
        """Push current prompt state for one object into the runner."""
        if self._runner and object_id in self._objects:
            prompts = self._objects[object_id].prompts
            prompts.frame_idx = self._frame_idx
            self._runner.set_prompts(object_id, prompts)

    def _schedule_inference(self) -> None:
        """Restart the debounce timer (80 ms)."""
        self._debounce_timer.start()

    def _advance_spinner(self) -> None:
        self._spin_frame = (self._spin_frame + 1) % 8
        self.update()

    # ── Coordinate transforms ──────────────────────────────────────────────

    def _frame_rect(self) -> QRectF:
        """
        Return the QRectF of the letterboxed video frame inside the widget.

        The frame is centred, aspect-ratio locked, and scaled by _zoom.
        """
        if self._frame_pixmap is None:
            return QRectF(self.rect())

        pw = self._frame_pixmap.width()
        ph = self._frame_pixmap.height()
        cw = self.width()
        ch = self.height()

        scale = min(cw / pw, ch / ph) * self._zoom
        fw = pw * scale
        fh = ph * scale

        x = (cw - fw) / 2 + self._pan_offset.x()
        y = (ch - fh) / 2 + self._pan_offset.y()
        return QRectF(x, y, fw, fh)

    def _canvas_to_video(self, p: QPointF) -> QPointF:
        """Map canvas (widget) coordinates to video pixel coordinates."""
        fr = self._frame_rect()
        if fr.width() == 0 or fr.height() == 0:
            return QPointF(0, 0)
        if self._frame_pixmap is None:
            return QPointF(0, 0)
        vx = (p.x() - fr.x()) / fr.width() * self._frame_pixmap.width()
        vy = (p.y() - fr.y()) / fr.height() * self._frame_pixmap.height()
        return QPointF(vx, vy)

    def _video_to_canvas(self, p: QPointF) -> QPointF:
        """Map video pixel coordinates to canvas (widget) coordinates."""
        fr = self._frame_rect()
        if self._frame_pixmap is None:
            return p
        cx = fr.x() + p.x() / self._frame_pixmap.width() * fr.width()
        cy = fr.y() + p.y() / self._frame_pixmap.height() * fr.height()
        return QPointF(cx, cy)

    def _clamp_video(self, vp: QPointF) -> QPointF:
        """Clamp a video-space point to the frame boundaries."""
        if self._frame_pixmap is None:
            return vp
        x = max(0.0, min(vp.x(), self._frame_pixmap.width() - 1))
        y = max(0.0, min(vp.y(), self._frame_pixmap.height() - 1))
        return QPointF(x, y)

    # ── Cursor ─────────────────────────────────────────────────────────────

    def _update_cursor(self) -> None:
        cursors = {
            "pan":    Qt.CursorShape.OpenHandCursor,
            "points": Qt.CursorShape.CrossCursor,
            "box":    Qt.CursorShape.CrossCursor,
            "mask":   Qt.CursorShape.ArrowCursor,
            "text":   Qt.CursorShape.ArrowCursor,
        }
        self.setCursor(cursors.get(self._mode, Qt.CursorShape.CrossCursor))

    # ── Mouse events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.position())
        btn = event.button()
        mod = event.modifiers()

        # Middle click → start pan
        if btn == Qt.MouseButton.MiddleButton:
            self._pan_last = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        fr = self._frame_rect()
        vp = self._clamp_video(self._canvas_to_video(pos))

        if self._mode == "pan":
            self._pan_last = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self._mode == "points":
            obj = self._active_obj()
            if obj is None:
                return
            label = 0 if btn == Qt.MouseButton.RightButton else 1
            pt = PromptPoint(x=vp.x(), y=vp.y(), label=label)

            # Check if clicking an existing point (within POINT_RADIUS)
            for i, existing in enumerate(obj.prompts.points):
                ep = self._video_to_canvas(QPointF(existing.x, existing.y))
                if (ep - pos).manhattanLength() < POINT_RADIUS * 2.5:
                    # Delete it
                    removed = obj.prompts.points.pop(i)
                    self._undo_stack.append(("point_delete", obj.object_id, removed))
                    self._sync_runner_prompts(obj.object_id)
                    self._schedule_inference()
                    self.prompts_changed.emit(obj.object_id)
                    self.update()
                    return

            obj.prompts.points.append(pt)
            self._undo_stack.append(("point", obj.object_id, pt))
            self._sync_runner_prompts(obj.object_id)
            self._schedule_inference()
            self.prompts_changed.emit(obj.object_id)
            self.update()
            return

        if self._mode == "box":
            self._box_start = vp
            self._box_end = vp
            self._box_drawing = True
            self.update()
            return

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.position())

        # Pan drag
        if self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_offset += QPointF(delta.x(), delta.y())
            self._pan_last = event.pos()
            self.update()
            return

        # Box drag
        if self._box_drawing and self._box_start is not None:
            vp = self._clamp_video(self._canvas_to_video(pos))
            self._box_end = vp
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        btn = event.button()
        pos = QPointF(event.position())

        if btn == Qt.MouseButton.MiddleButton or (
            self._mode == "pan" and self._pan_last is not None
        ):
            self._pan_last = None
            self._update_cursor()
            return

        if self._box_drawing and self._mode == "box" and self._box_start is not None:
            vp = self._clamp_video(self._canvas_to_video(pos))
            self._box_end = vp
            self._box_drawing = False

            x1 = min(self._box_start.x(), self._box_end.x())
            y1 = min(self._box_start.y(), self._box_end.y())
            x2 = max(self._box_start.x(), self._box_end.x())
            y2 = max(self._box_start.y(), self._box_end.y())

            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                obj = self._active_obj()
                if obj:
                    obj.prompts.box = PromptBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    self._undo_stack.append(("box", obj.object_id))
                    self._sync_runner_prompts(obj.object_id)
                    self._schedule_inference()
                    self.prompts_changed.emit(obj.object_id)

            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_zoom = max(0.1, min(self._zoom * factor, 8.0))

        # Zoom toward cursor position
        cursor_pos = QPointF(event.position())
        old_zoom = self._zoom
        self._zoom = new_zoom
        zoom_ratio = new_zoom / old_zoom
        self._pan_offset = cursor_pos + (self._pan_offset - cursor_pos) * zoom_ratio

        self.update()
        self.inference_requested.emit()   # re-request zoom % update (no debounce needed)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            self.undo_last_prompt()
            return

        mode_keys = {
            Qt.Key.Key_P: "points",
            Qt.Key.Key_B: "box",
            Qt.Key.Key_M: "mask",
            Qt.Key.Key_T: "text",
        }
        if key in mode_keys:
            self.set_mode(mode_keys[key])
            return

        super().keyPressEvent(event)

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1C1C1C"))

        if self._frame_pixmap is None:
            self._draw_placeholder(painter)
            return

        fr = self._frame_rect()

        # 1. Video frame
        painter.drawPixmap(fr.toRect(), self._frame_pixmap)

        # 2. Mask overlays (per object, sorted by ID)
        for obj_id in sorted(self._objects.keys()):
            obj = self._objects[obj_id]
            if not obj.visible or obj.mask is None:
                continue
            self._draw_mask_overlay(painter, obj, fr)

        # 3. Box being drawn (dashed) or committed (solid)
        self._draw_box(painter, fr)

        # 4. Prompt points
        for obj_id in sorted(self._objects.keys()):
            obj = self._objects[obj_id]
            self._draw_points(painter, obj, fr)

        # 5. Frame counter overlay
        self._draw_frame_counter(painter)

        # 6. Inference spinner
        if self._inference_running:
            self._draw_spinner(painter)

        painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor("#555555"))
        painter.setFont(QFont("system-ui", 14))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Select a clip to begin",
        )

    def _draw_mask_overlay(
        self, painter: QPainter, obj: ObjectState, fr: QRectF
    ) -> None:
        if obj.mask is None:
            return
        mask = obj.mask   # (H, W) uint8

        # Resize mask to frame_rect dimensions for drawing
        fw, fh = int(fr.width()), int(fr.height())
        if fw <= 0 or fh <= 0:
            return

        resized = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        # Build RGBA image: object color at user opacity where mask > 0
        c = obj.color
        alpha = int(obj.opacity / 100 * 180)   # max 180/255 so frame shows through

        rgba = np.zeros((fh, fw, 4), dtype=np.uint8)
        where = resized > 0
        rgba[where, 0] = c.red()
        rgba[where, 1] = c.green()
        rgba[where, 2] = c.blue()
        rgba[where, 3] = alpha

        img = QImage(
            rgba.tobytes(), fw, fh, fw * 4,
            QImage.Format.Format_RGBA8888,
        )
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )
        painter.drawImage(QPoint(int(fr.x()), int(fr.y())), img)

        # White contour (1.5px)
        self._draw_contour(painter, resized, fr)

    def _draw_contour(
        self, painter: QPainter, mask_resized: np.ndarray, fr: QRectF
    ) -> None:
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return

        pen = QPen(QColor(255, 255, 255, 200), 1.5)
        painter.setPen(pen)
        ox, oy = int(fr.x()), int(fr.y())

        for contour in contours:
            pts = contour.squeeze(axis=1)
            if pts.ndim < 2 or len(pts) < 2:
                continue
            from PyQt6.QtCore import QLine
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                painter.drawLine(
                    ox + int(p1[0]), oy + int(p1[1]),
                    ox + int(p2[0]), oy + int(p2[1]),
                )

    def _draw_points(
        self, painter: QPainter, obj: ObjectState, fr: QRectF
    ) -> None:
        is_active = (obj.object_id == self._active_object_id)

        for i, pt in enumerate(obj.prompts.points):
            cp = self._video_to_canvas(QPointF(pt.x, pt.y))

            fill = QColor("#00AA00") if pt.label == 1 else QColor("#CC0000")
            pen_color = QColor("#FFFFFF")

            # Outer white ring
            painter.setPen(QPen(pen_color, 1.5))
            painter.setBrush(fill)
            painter.drawEllipse(cp, POINT_RADIUS, POINT_RADIUS)

            # Number label
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("system-ui", POINT_FONT_SIZE, QFont.Weight.Bold))
            painter.drawText(
                QRectF(cp.x() - 10, cp.y() - 10, 20, 20),
                Qt.AlignmentFlag.AlignCenter,
                str(i + 1),
            )

    def _draw_box(self, painter: QPainter, fr: QRectF) -> None:
        obj = self._active_obj()
        if obj is None:
            return

        # In-progress box (dashed)
        if self._box_drawing and self._box_start and self._box_end:
            c1 = self._video_to_canvas(self._box_start)
            c2 = self._video_to_canvas(self._box_end)
            pen = QPen(QColor(COLOR_ACCENT), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(c1, c2))

        # Committed box (solid) from active object prompts
        box = obj.prompts.box
        if box and not self._box_drawing:
            c1 = self._video_to_canvas(QPointF(box.x1, box.y1))
            c2 = self._video_to_canvas(QPointF(box.x2, box.y2))
            pen = QPen(QColor(COLOR_ACCENT), 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(c1, c2))

    def _draw_frame_counter(self, painter: QPainter) -> None:
        text = f"Frame {self._frame_idx + 1:05d} / {self._total_frames:05d}"
        painter.setFont(QFont("system-ui", 10))
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        h = fm.height() + 6
        x = 8
        y = self.height() - h - 8
        painter.fillRect(x, y, w, h, QColor(0, 0, 0, 160))
        painter.setPen(QColor("#E8E8E8"))
        painter.drawText(x + 6, y + h - 5, text)

    def _draw_spinner(self, painter: QPainter) -> None:
        """Draw a small spinning arc indicator in the top-right corner."""
        import math
        cx = self.width() - 20
        cy = 20
        r = 8
        span = 270
        start = self._spin_frame * 45

        pen = QPen(QColor(COLOR_ACCENT), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            QRect(cx - r, cy - r, r * 2, r * 2),
            start * 16,
            span * 16,
        )

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert an OpenCV BGR frame to a QPixmap."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img)
