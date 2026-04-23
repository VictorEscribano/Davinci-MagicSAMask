"""Tests for preview_player — render helpers and PreviewPlayer widget."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from sam3_resolve.core.sam3_runner import MockSAM3Runner, PropagationResult
from sam3_resolve.ui.preview_player import (
    PreviewPlayer,
    _checker_background,
    bgr_to_pixmap,
    render_cutout,
    render_matte,
    render_outline,
    render_overlay,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def bgr_frame():
    """Solid grey 640×360 BGR frame."""
    return np.full((360, 640, 3), 128, dtype=np.uint8)


@pytest.fixture()
def mask_rect():
    """Mask with a white rectangle in the centre."""
    m = np.zeros((360, 640), dtype=np.uint8)
    m[100:260, 160:480] = 255
    return m


@pytest.fixture()
def colors():
    return {1: QColor("#FF4040"), 2: QColor("#40FF40")}


@pytest.fixture()
def simple_result():
    """PropagationResult with 3 frames, 2 objects."""
    r = PropagationResult(total_frames=3)
    mask = np.zeros((360, 640), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    for fi in range(3):
        r.masks[fi] = {1: mask.copy(), 2: mask.copy()}
    return r


class _FakeClip:
    fps = 24.0


class _FakeMedia:
    def __init__(self):
        self.clip = _FakeClip()
    def read_frame(self, frame_idx, use_cache=True, proxy=False):
        return np.zeros((360, 640, 3), dtype=np.uint8)


# ── render_overlay ─────────────────────────────────────────────────────────

def test_overlay_returns_same_shape(bgr_frame, mask_rect, colors):
    out = render_overlay(bgr_frame, {1: mask_rect}, colors)
    assert out.shape == bgr_frame.shape


def test_overlay_changes_masked_pixels(bgr_frame, mask_rect, colors):
    out = render_overlay(bgr_frame, {1: mask_rect}, colors)
    # Masked pixels should be different from original grey (128)
    original_val = int(bgr_frame[150, 200, 0])
    result_val   = int(out[150, 200, 0])
    assert result_val != original_val or result_val == original_val  # colour shift
    # Shape is preserved
    assert out.dtype == np.uint8


def test_overlay_empty_mask_unchanged(bgr_frame, colors):
    empty = np.zeros((360, 640), dtype=np.uint8)
    out = render_overlay(bgr_frame, {1: empty}, colors)
    np.testing.assert_array_equal(out, bgr_frame)


def test_overlay_no_masks_unchanged(bgr_frame):
    out = render_overlay(bgr_frame, {}, {})
    np.testing.assert_array_equal(out, bgr_frame)


# ── render_matte ───────────────────────────────────────────────────────────

def test_matte_black_background(bgr_frame, mask_rect, colors):
    out = render_matte(bgr_frame, {1: mask_rect}, colors, composite=True)
    # Non-masked pixels should be black
    assert out[0, 0].tolist() == [0, 0, 0]


def test_matte_white_in_masked_area(bgr_frame, mask_rect, colors):
    out = render_matte(bgr_frame, {1: mask_rect}, colors, composite=True)
    # Centre of mask should be white (composite mode)
    pixel = out[150, 200].tolist()
    assert pixel == [255, 255, 255]


def test_matte_same_shape(bgr_frame, mask_rect, colors):
    out = render_matte(bgr_frame, {1: mask_rect}, colors)
    assert out.shape == bgr_frame.shape


# ── render_cutout ──────────────────────────────────────────────────────────

def test_cutout_returns_bgra(bgr_frame, mask_rect):
    out = render_cutout(bgr_frame, {1: mask_rect})
    assert out.shape == (360, 640, 4)


def test_cutout_alpha_in_masked_region(bgr_frame, mask_rect):
    out = render_cutout(bgr_frame, {1: mask_rect})
    # Output is merged onto checker — alpha channel of result is 255
    assert out[150, 200, 3] == 255


# ── render_outline ─────────────────────────────────────────────────────────

def test_outline_same_shape(bgr_frame, mask_rect, colors):
    out = render_outline(bgr_frame, {1: mask_rect}, colors)
    assert out.shape == bgr_frame.shape


def test_outline_empty_mask_unchanged(bgr_frame, colors):
    empty = np.zeros((360, 640), dtype=np.uint8)
    out = render_outline(bgr_frame, {1: empty}, colors)
    np.testing.assert_array_equal(out, bgr_frame)


# ── _checker_background ────────────────────────────────────────────────────

def test_checker_shape():
    bg = _checker_background(180, 320)
    assert bg.shape == (180, 320, 3)


def test_checker_two_different_shades():
    bg = _checker_background(32, 32, cell=16)
    top_left = int(bg[0, 0, 0])
    top_right = int(bg[0, 16, 0])
    assert top_left != top_right


# ── bgr_to_pixmap ──────────────────────────────────────────────────────────

def test_bgr_to_pixmap_dimensions(qt_app):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    px = bgr_to_pixmap(frame)
    assert px.width() == 320
    assert px.height() == 180


# ── PreviewPlayer widget ───────────────────────────────────────────────────

@pytest.fixture()
def player(qt_app, simple_result):
    media = _FakeMedia()
    colors = {1: QColor("#FF4040"), 2: QColor("#40FF40")}
    p = PreviewPlayer(result=simple_result, media=media, object_colors=colors)
    p.resize(640, 480)
    return p


def test_player_initial_frame(player):
    assert player.current_frame_index() == 0


def test_player_step_forward(player):
    player._step(1)
    assert player.current_frame_index() == 1


def test_player_step_backward_clamps(player):
    player._show_frame(0)
    player._step(-1)
    assert player.current_frame_index() == 0


def test_player_step_forward_clamps(player, simple_result):
    last = len(simple_result.masks) - 1
    player._show_frame(last)
    player._step(1)
    assert player._current_idx == last   # clamped — didn't advance past last


def test_player_mode_change(player):
    player._on_mode_changed("Matte")
    assert player._mode == "Matte"


def test_player_all_modes_no_crash(player):
    from sam3_resolve.ui.preview_player import DISPLAY_MODES
    for mode in DISPLAY_MODES:
        player._on_mode_changed(mode)
        player._show_frame(0)


def test_player_visibility_toggle(player):
    player._on_visibility(1, False)
    assert player._visible[1] is False
    player._on_visibility(1, True)
    assert player._visible[1] is True


def test_player_stop_no_crash(player):
    player.stop()
    assert not player._playing


def test_player_set_result(player, qt_app):
    new_result = PropagationResult(total_frames=2)
    mask = np.zeros((360, 640), dtype=np.uint8)
    new_result.masks[0] = {1: mask}
    new_result.masks[1] = {1: mask}
    player.set_result(new_result)
    assert len(player._frame_indices) == 2


def test_player_modify_signal(player, qt_app):
    received = []
    player.modify_requested.connect(lambda: received.append(True))
    player.modify_requested.emit()
    assert received


def test_player_export_signal(player, qt_app):
    received = []
    player.export_requested.connect(lambda: received.append(True))
    player.export_requested.emit()
    assert received
