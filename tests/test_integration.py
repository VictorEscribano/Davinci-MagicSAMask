"""
Full integration test — Step 15.

Wires together:
  MockResolveBridge → ClipInfo
  MockSAM3Runner    → propagation with synthetic ellipse masks
  CanvasWidget      → prompt state management
  MaskExporter      → 16-bit PNG output with manifest
  PreviewPlayer     → render modes on propagation result

All components run synchronously (no real QThread.start()) so the test
suite stays deterministic and fast.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from sam3_resolve.core.mask_exporter import ExportSettings, MaskExporter
from sam3_resolve.core.resolve_bridge import MockResolveBridge
from sam3_resolve.core.sam3_runner import (
    MockSAM3Runner,
    ObjectPrompts,
    PropagationResult,
    PromptBox,
    PromptPoint,
)
from sam3_resolve.ui.canvas_widget import CanvasWidget
from sam3_resolve.ui.preview_player import (
    PreviewPlayer,
    render_matte,
    render_overlay,
    render_outline,
)
from sam3_resolve.ui.workers import LiveInferenceWorker, PropagationWorker


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def runner():
    r = MockSAM3Runner()
    r._load_delay_ms = 0.0
    r.load_model()
    return r


@pytest.fixture(scope="module")
def bridge():
    return MockResolveBridge()


@pytest.fixture()
def canvas(qt_app, runner):
    w = CanvasWidget(runner=runner)
    w.resize(800, 450)
    w.add_object(1)
    w.set_active_object(1)
    return w


@pytest.fixture()
def frame_360p():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)


# ── 1. Bridge → clip info ──────────────────────────────────────────────────

def test_bridge_returns_clip(bridge):
    clip = bridge.get_current_clip()
    assert clip is not None
    assert clip.name != ""


def test_bridge_clip_has_resolution(bridge):
    clip = bridge.get_current_clip()
    assert clip.width > 0
    assert clip.height > 0


def test_bridge_clip_fps(bridge):
    clip = bridge.get_current_clip()
    assert clip.fps > 0


# ── 2. Runner prompt → single-frame inference ──────────────────────────────

def test_runner_single_frame_with_point(runner, frame_360p):
    runner.clear_all_prompts()
    prompts = ObjectPrompts(object_id=1)
    prompts.points.append(PromptPoint(x=320, y=180, label=1))
    runner.set_prompts(1, prompts)
    masks = runner.run_single_frame(frame_idx=0, frame=frame_360p)
    assert 1 in masks
    assert masks[1].shape == (360, 640)


def test_runner_single_frame_with_box(runner, frame_360p):
    runner.clear_all_prompts()
    prompts = ObjectPrompts(object_id=1)
    prompts.box = PromptBox(100, 80, 540, 280)
    runner.set_prompts(1, prompts)
    masks = runner.run_single_frame(0, frame_360p)
    assert masks[1].max() == 255


def test_runner_single_frame_no_prompt_empty(runner, frame_360p):
    runner.clear_all_prompts()
    runner.set_prompts(1, ObjectPrompts(object_id=1))  # no points or box
    masks = runner.run_single_frame(0, frame_360p)
    assert 1 not in masks


# ── 3. Canvas ↔ runner round-trip ─────────────────────────────────────────

def test_canvas_prompts_fed_to_runner(canvas, frame_360p, runner):
    canvas.set_frame(frame_360p, 0, 10)
    obj = canvas._active_obj()
    obj.prompts.points.append(PromptPoint(320, 180, 1))
    canvas._sync_runner_prompts(1)
    # Runner should now have prompts for object 1
    p = runner.get_prompts(1)
    assert len(p.points) >= 1


def test_canvas_clear_removes_runner_prompts(canvas, runner):
    canvas.clear_all_prompts()
    p = runner.get_prompts(1)
    assert len(p.points) == 0
    assert p.box is None


# ── 4. LiveInferenceWorker ─────────────────────────────────────────────────

def test_live_worker_emits_masks(qt_app, runner, frame_360p):
    runner.clear_all_prompts()
    prompts = ObjectPrompts(object_id=1)
    prompts.points.append(PromptPoint(320, 180, 1))
    runner.set_prompts(1, prompts)

    worker = LiveInferenceWorker(runner)
    worker.set_task(frame_idx=0, frame=frame_360p)

    received = []
    worker.mask_ready.connect(lambda m: received.append(m))

    worker.run()   # synchronous — no .start()
    assert len(received) == 1
    assert 1 in received[0]


def test_live_worker_no_frame_no_emit(qt_app, runner):
    worker = LiveInferenceWorker(runner)
    # Don't call set_task
    received = []
    worker.mask_ready.connect(lambda m: received.append(m))
    worker.run()
    assert received == []


# ── 5. PropagationWorker (synthetic frames) ───────────────────────────────

class _FakeClip:
    in_point_frame  = 0
    out_point_frame = 5
    fps = 24.0


class _FakeMedia:
    def __init__(self):
        self.clip = _FakeClip()

    def iter_frames(self, start, end, stop_event=None):
        rng = np.random.default_rng(42)
        for fi in range(start, end):
            if stop_event and stop_event.is_set():
                return
            yield fi, rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)


def test_propagation_worker_produces_masks(qt_app, runner):
    runner.clear_all_prompts()
    prompts = ObjectPrompts(object_id=1)
    prompts.points.append(PromptPoint(320, 180, 1))
    runner.set_prompts(1, prompts)

    media = _FakeMedia()
    worker = PropagationWorker(runner=runner, media_handler=media)

    results = []
    worker.finished.connect(lambda r: results.append(r))

    worker.run()   # synchronous
    assert len(results) == 1
    result = results[0]
    assert not result.cancelled
    assert len(result.masks) == 5   # frames 0–4


def test_propagation_worker_cancel(qt_app, runner):
    runner.clear_all_prompts()
    prompts = ObjectPrompts(object_id=1)
    prompts.points.append(PromptPoint(320, 180, 1))
    runner.set_prompts(1, prompts)

    media = _FakeMedia()
    worker = PropagationWorker(runner=runner, media_handler=media)
    worker.cancel()   # pre-cancel

    cancelled = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run()
    assert cancelled


# ── 6. Mask export end-to-end ─────────────────────────────────────────────

def _make_result(n_frames: int, n_objects: int = 1) -> PropagationResult:
    r = PropagationResult(total_frames=n_frames)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    for fi in range(n_frames):
        r.masks[fi] = {oid: mask.copy() for oid in range(1, n_objects + 1)}
    return r


def test_export_creates_pngs(tmp_path):
    result = _make_result(3)
    settings = ExportSettings(
        output_dir=tmp_path,
        original_width=100,
        original_height=100,
    )
    export_result = MaskExporter(result, settings).export()
    assert export_result.success
    assert export_result.frames_written == 3
    pngs = list((tmp_path / "object_001").glob("*.png"))
    assert len(pngs) == 3


def test_export_pngs_are_16bit(tmp_path):
    result = _make_result(1)
    settings = ExportSettings(output_dir=tmp_path, original_width=100, original_height=100)
    MaskExporter(result, settings).export()
    png = next((tmp_path / "object_001").glob("*.png"))
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16


def test_export_manifest_lists_objects(tmp_path):
    result = _make_result(2, n_objects=2)
    settings = ExportSettings(
        output_dir=tmp_path,
        original_width=100,
        original_height=100,
        object_names={1: "Hero", 2: "Shadow"},
    )
    MaskExporter(result, settings).export()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    names = [o["name"] for o in manifest["objects"]]
    assert "Hero" in names
    assert "Shadow" in names


# ── 7. PreviewPlayer render modes ─────────────────────────────────────────

@pytest.fixture()
def prop_result():
    return _make_result(3)


@pytest.fixture()
def player_colors():
    return {1: QColor("#FF4040")}


class _MediaStub:
    class clip:
        fps = 24.0
    def read_frame(self, fi, **kw):
        return np.zeros((100, 100, 3), dtype=np.uint8)


def test_player_overlay_no_crash(qt_app, prop_result, player_colors):
    p = PreviewPlayer(prop_result, _MediaStub(), player_colors)
    p._on_mode_changed("Overlay")
    p._show_frame(0)


def test_player_matte_no_crash(qt_app, prop_result, player_colors):
    p = PreviewPlayer(prop_result, _MediaStub(), player_colors)
    p._on_mode_changed("Matte")
    p._show_frame(0)


def test_player_cutout_no_crash(qt_app, prop_result, player_colors):
    p = PreviewPlayer(prop_result, _MediaStub(), player_colors)
    p._on_mode_changed("Cutout")
    p._show_frame(0)


def test_player_outline_no_crash(qt_app, prop_result, player_colors):
    p = PreviewPlayer(prop_result, _MediaStub(), player_colors)
    p._on_mode_changed("Outline")
    p._show_frame(0)


def test_player_frame_count(qt_app, prop_result, player_colors):
    p = PreviewPlayer(prop_result, _MediaStub(), player_colors)
    assert len(p._frame_indices) == 3


# ── 8. Full pipeline: prompts → propagation → export ──────────────────────

def test_full_pipeline(tmp_path, runner):
    """End-to-end: set prompt → propagate 5 frames → export 5 PNGs."""
    runner.clear_all_prompts()
    runner._load_delay_ms = 0.0

    # Set up a positive point prompt
    prompts = ObjectPrompts(object_id=1)
    prompts.points.append(PromptPoint(x=320, y=180, label=1))
    runner.set_prompts(1, prompts)

    # Propagate
    media = _FakeMedia()
    prev_call_count = len(runner.propagation_calls)
    worker = PropagationWorker(runner=runner, media_handler=media)

    results = []
    worker.finished.connect(lambda r: results.append(r))
    worker.run()

    assert len(results) == 1
    result = results[0]
    assert not result.cancelled
    assert len(runner.propagation_calls) == prev_call_count + 1

    # Export
    settings = ExportSettings(
        output_dir=tmp_path,
        original_width=640,
        original_height=360,
        object_names={1: "Subject"},
    )
    exp = MaskExporter(result, settings).export()
    assert exp.success
    assert exp.frames_written == 5

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["objects"][0]["name"] == "Subject"
    assert manifest["objects"][0]["frame_count"] == 5
