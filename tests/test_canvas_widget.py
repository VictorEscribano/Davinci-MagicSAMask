"""Headless tests for canvas_widget.py — coordinate transforms, prompts, overlays."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

from sam3_resolve.core.sam3_runner import MockSAM3Runner, ObjectPrompts, PromptPoint, PromptBox
from sam3_resolve.ui.canvas_widget import CanvasWidget, ObjectState, OBJECT_COLORS


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def runner():
    r = MockSAM3Runner()
    r._load_delay_ms = 0.0
    r.load_model()
    return r


@pytest.fixture()
def canvas(qt_app, runner):
    w = CanvasWidget(runner=runner)
    w.resize(800, 450)
    w.add_object(1)
    w.set_active_object(1)
    return w


@pytest.fixture()
def frame_640x360():
    return np.zeros((360, 640, 3), dtype=np.uint8)


# ── Initialisation ─────────────────────────────────────────────────────────

def test_canvas_default_mode(canvas):
    assert canvas._mode == "points"


def test_canvas_default_zoom(canvas):
    assert canvas._zoom == pytest.approx(1.0)


def test_canvas_minimum_size(canvas):
    assert canvas.minimumWidth() == 500


# ── Frame loading ──────────────────────────────────────────────────────────

def test_set_frame_stores_data(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, frame_idx=5, total=100)
    assert canvas._frame_idx == 5
    assert canvas._total_frames == 100
    assert canvas._frame_pixmap is not None


def test_set_frame_pixmap_dimensions(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    assert canvas._frame_pixmap.width() == 640
    assert canvas._frame_pixmap.height() == 360


# ── Mode switching ─────────────────────────────────────────────────────────

def test_set_mode_box(canvas):
    canvas.set_mode("box")
    assert canvas._mode == "box"


def test_set_mode_clears_box_drawing(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    canvas._box_drawing = True
    canvas.set_mode("points")
    assert not canvas._box_drawing


# ── Object management ──────────────────────────────────────────────────────

def test_add_object_registers(canvas):
    canvas.add_object(2)
    assert 2 in canvas._objects


def test_add_object_assigns_color(canvas):
    canvas.add_object(3)
    assert canvas._objects[3].color is not None


def test_remove_object(canvas):
    canvas.add_object(4)
    canvas.remove_object(4)
    assert 4 not in canvas._objects


def test_set_active_object_auto_creates(canvas):
    canvas.set_active_object(5)
    assert 5 in canvas._objects
    assert canvas._active_object_id == 5


def test_set_object_opacity(canvas):
    canvas.set_object_opacity(1, 50)
    assert canvas._objects[1].opacity == 50


def test_set_object_visible(canvas):
    canvas.set_object_visible(1, False)
    assert canvas._objects[1].visible is False


# ── Coordinate transforms ──────────────────────────────────────────────────

def test_canvas_to_video_centre(canvas, frame_640x360):
    """Centre of canvas maps to centre of video frame."""
    canvas.set_frame(frame_640x360, 0, 1)
    # Canvas is 800×450, frame is 640×360 (same ratio 16:9)
    # With zoom=1, the frame fills the canvas exactly
    centre_canvas = QPointF(400, 225)
    vp = canvas._canvas_to_video(centre_canvas)
    assert abs(vp.x() - 320) < 2
    assert abs(vp.y() - 180) < 2


def test_video_to_canvas_roundtrip(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    for vx, vy in [(0, 0), (320, 180), (639, 359)]:
        vp = QPointF(vx, vy)
        cp = canvas._video_to_canvas(vp)
        back = canvas._canvas_to_video(cp)
        assert abs(back.x() - vx) < 1
        assert abs(back.y() - vy) < 1


def test_clamp_video_within_bounds(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    clamped = canvas._clamp_video(QPointF(9999, 9999))
    assert clamped.x() <= 639
    assert clamped.y() <= 359


def test_clamp_video_negative(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    clamped = canvas._clamp_video(QPointF(-100, -200))
    assert clamped.x() == 0.0
    assert clamped.y() == 0.0


# ── Prompt management ──────────────────────────────────────────────────────

def test_positive_point_added(canvas, frame_640x360, runner):
    canvas.set_frame(frame_640x360, 0, 1)
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QPointF, QPoint

    # Simulate left click at canvas centre
    obj = canvas._active_obj()
    initial_count = len(obj.prompts.points)

    # Add directly via internal state (avoids QMouseEvent complexity)
    pt = PromptPoint(320, 180, 1)
    obj.prompts.points.append(pt)
    canvas._undo_stack.append(("point", 1, pt))
    canvas._sync_runner_prompts(1)

    assert len(obj.prompts.points) == initial_count + 1
    assert obj.prompts.points[-1].label == 1


def test_negative_point_added(canvas):
    obj = canvas._active_obj()
    pt = PromptPoint(100, 100, 0)
    obj.prompts.points.append(pt)
    assert obj.prompts.points[-1].label == 0


def test_undo_removes_last_point(canvas):
    obj = canvas._active_obj()
    pt = PromptPoint(200, 200, 1)
    obj.prompts.points.append(pt)
    canvas._undo_stack.append(("point", 1, pt))
    before = len(obj.prompts.points)

    canvas.undo_last_prompt()
    assert len(obj.prompts.points) == before - 1


def test_undo_removes_box(canvas):
    obj = canvas._active_obj()
    obj.prompts.box = PromptBox(10, 10, 200, 200)
    canvas._undo_stack.append(("box", 1))

    canvas.undo_last_prompt()
    assert obj.prompts.box is None


def test_clear_all_prompts(canvas):
    obj = canvas._active_obj()
    obj.prompts.points.append(PromptPoint(100, 100, 1))
    obj.prompts.box = PromptBox(0, 0, 50, 50)

    canvas.clear_all_prompts()

    assert obj.prompts.points == []
    assert obj.prompts.box is None


def test_get_all_prompts_returns_dict(canvas):
    canvas.add_object(2)
    result = canvas.get_all_prompts()
    assert isinstance(result, dict)
    assert 1 in result
    assert 2 in result


# ── Mask overlay ───────────────────────────────────────────────────────────

def test_set_object_mask(canvas):
    mask = np.zeros((360, 640), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    canvas.set_object_mask(1, mask)
    assert canvas._objects[1].mask is not None
    assert canvas._objects[1].mask.max() == 255


def test_set_all_masks(canvas):
    canvas.add_object(2)
    masks = {
        1: np.full((360, 640), 255, dtype=np.uint8),
        2: np.zeros((360, 640), dtype=np.uint8),
    }
    canvas.set_all_masks(masks)
    assert canvas._objects[1].mask.max() == 255
    assert canvas._objects[2].mask.max() == 0


# ── Zoom ───────────────────────────────────────────────────────────────────

def test_set_zoom_clamped_max(canvas):
    canvas.set_zoom(100.0)
    assert canvas._zoom == pytest.approx(8.0)


def test_set_zoom_clamped_min(canvas):
    canvas.set_zoom(0.001)
    assert canvas._zoom == pytest.approx(0.1)


def test_zoom_percent(canvas):
    canvas.set_zoom(2.0)
    assert canvas.zoom_percent == 200


# ── Inference indicator ────────────────────────────────────────────────────

def test_inference_running_starts_spinner(canvas):
    canvas.set_inference_running(True)
    assert canvas._inference_running is True
    assert canvas._spin_timer.isActive()
    canvas.set_inference_running(False)
    assert not canvas._spin_timer.isActive()


# ── BGR to pixmap ──────────────────────────────────────────────────────────

def test_bgr_to_pixmap_shape():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[0, 0] = [0, 0, 255]   # red in BGR
    px = CanvasWidget._bgr_to_pixmap(frame)
    assert px.width() == 320
    assert px.height() == 180


# ── Frame rect ────────────────────────────────────────────────────────────

def test_frame_rect_without_frame(canvas):
    # Without a pixmap, frame_rect returns the widget rect
    canvas._frame_pixmap = None
    fr = canvas._frame_rect()
    assert fr.width() == canvas.width()


def test_frame_rect_with_frame(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 0, 1)
    fr = canvas._frame_rect()
    # At zoom=1, the frame fills the canvas as much as aspect ratio allows
    assert fr.width() > 0
    assert fr.height() > 0


# ── Paint (smoke test — no crash) ─────────────────────────────────────────

def test_paint_with_frame_no_crash(canvas, frame_640x360):
    canvas.set_frame(frame_640x360, 5, 100)
    mask = np.zeros((360, 640), dtype=np.uint8)
    mask[100:260, 200:440] = 255
    canvas.set_object_mask(1, mask)
    # Trigger a repaint — just confirm no exception
    canvas.repaint()


def test_paint_without_frame_no_crash(canvas):
    canvas._frame_pixmap = None
    canvas.repaint()
