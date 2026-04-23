"""Tests for sam3_runner — all using MockSAM3Runner (no GPU required)."""

from __future__ import annotations

import threading
import time
from typing import Iterator

import cv2
import numpy as np
import pytest

from sam3_resolve.core.sam3_runner import (
    MockSAM3Runner,
    ObjectPrompts,
    PromptBox,
    PromptPoint,
    PropagationResult,
    confidence_level,
    create_runner,
    mask_is_empty,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def runner() -> MockSAM3Runner:
    r = MockSAM3Runner()
    r._load_delay_ms = 0.0   # disable simulated latency in tests
    r.load_model()
    return r


@pytest.fixture()
def frame_320x180() -> np.ndarray:
    return np.zeros((180, 320, 3), dtype=np.uint8)


@pytest.fixture()
def frames_10(frame_320x180) -> list[tuple[int, np.ndarray]]:
    return [(i, frame_320x180.copy()) for i in range(10)]


def _iter(frames: list[tuple[int, np.ndarray]]) -> Iterator[tuple[int, np.ndarray]]:
    return iter(frames)


# ── Lifecycle ──────────────────────────────────────────────────────────────

def test_load_model_sets_loaded():
    r = MockSAM3Runner()
    assert not r.is_loaded()
    r.load_model()
    assert r.is_loaded()


def test_unload_model_clears_loaded(runner):
    runner.unload_model()
    assert not runner.is_loaded()


# ── Prompt management ──────────────────────────────────────────────────────

def test_set_and_get_prompts(runner):
    prompts = ObjectPrompts(
        object_id=1,
        points=[PromptPoint(100, 90, 1)],
    )
    runner.set_prompts(1, prompts)
    retrieved = runner.get_prompts(1)
    assert retrieved.object_id == 1
    assert len(retrieved.points) == 1
    assert retrieved.points[0].x == 100


def test_get_empty_prompts_returns_default(runner):
    p = runner.get_prompts(99)
    assert p.object_id == 99
    assert p.points == []
    assert not p.has_prompts()


def test_clear_object_prompts(runner):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(10, 10, 1)]))
    runner.set_prompts(2, ObjectPrompts(2, [PromptPoint(20, 20, 1)]))
    runner.clear_object_prompts(1)
    assert 1 not in runner.object_ids
    assert 2 in runner.object_ids


def test_clear_all_prompts(runner):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(10, 10, 1)]))
    runner.set_prompts(2, ObjectPrompts(2, [PromptPoint(20, 20, 1)]))
    runner.clear_all_prompts()
    assert runner.object_ids == []


def test_object_ids_reflects_prompt_state(runner):
    runner.set_prompts(3, ObjectPrompts(3, [PromptPoint(5, 5, 1)]))
    runner.set_prompts(7, ObjectPrompts(7, [PromptPoint(5, 5, 1)]))
    assert sorted(runner.object_ids) == [3, 7]


# ── ObjectPrompts helpers ──────────────────────────────────────────────────

def test_has_prompts_positive_point():
    p = ObjectPrompts(1, [PromptPoint(10, 10, 1)])
    assert p.has_prompts()


def test_has_prompts_negative_only_is_false():
    p = ObjectPrompts(1, [PromptPoint(10, 10, 0)])
    assert not p.has_prompts()


def test_has_prompts_box_only():
    p = ObjectPrompts(1, box=PromptBox(0, 0, 50, 50))
    assert p.has_prompts()


def test_points_array_shape():
    p = ObjectPrompts(1, [PromptPoint(10, 20, 1), PromptPoint(30, 40, 0)])
    arr = p.points_array()
    assert arr is not None
    assert arr.shape == (2, 2)
    assert arr[0, 0] == pytest.approx(10)


def test_labels_array_values():
    p = ObjectPrompts(1, [PromptPoint(10, 20, 1), PromptPoint(30, 40, 0)])
    labels = p.labels_array()
    assert list(labels) == [1, 0]


def test_prompt_box_numpy():
    box = PromptBox(10, 20, 110, 120)
    arr = box.as_numpy()
    assert list(arr) == pytest.approx([10, 20, 110, 120])


# ── Single-frame inference ─────────────────────────────────────────────────

def test_single_frame_returns_dict(runner, frame_320x180):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    masks = runner.run_single_frame(0, frame_320x180)
    assert 1 in masks
    assert masks[1].shape == (180, 320)


def test_single_frame_mask_has_content(runner, frame_320x180):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    masks = runner.run_single_frame(0, frame_320x180)
    assert masks[1].max() == 255


def test_single_frame_no_prompts_returns_empty(runner, frame_320x180):
    masks = runner.run_single_frame(0, frame_320x180)
    assert masks == {}


def test_single_frame_multiple_objects(runner, frame_320x180):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(80, 90, 1)]))
    runner.set_prompts(2, ObjectPrompts(2, [PromptPoint(240, 90, 1)]))
    masks = runner.run_single_frame(0, frame_320x180)
    assert set(masks.keys()) == {1, 2}


def test_single_frame_box_prompt(runner, frame_320x180):
    prompts = ObjectPrompts(1, box=PromptBox(50, 40, 270, 140))
    runner.set_prompts(1, prompts)
    masks = runner.run_single_frame(0, frame_320x180)
    assert 1 in masks
    assert not mask_is_empty(masks[1])


# ── Propagation ────────────────────────────────────────────────────────────

def test_propagation_returns_result(runner, frames_10):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    result = runner.propagate(_iter(frames_10), 0, 10)
    assert isinstance(result, PropagationResult)
    assert len(result.masks) == 10


def test_propagation_all_frames_present(runner, frames_10):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    result = runner.propagate(_iter(frames_10), 0, 10)
    for idx in range(10):
        assert idx in result.masks


def test_propagation_progress_callback(runner, frames_10):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    calls: list[tuple] = []
    runner.propagate(_iter(frames_10), 0, 10, progress_callback=lambda *a: calls.append(a))
    assert len(calls) == 10
    # Last call: processed=10, total=10
    assert calls[-1][0] == 10
    assert calls[-1][1] == 10


def test_propagation_mask_callback(runner, frames_10):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    received: list[tuple] = []
    runner.propagate(_iter(frames_10), 0, 10, mask_callback=lambda fi, m: received.append((fi, m)))
    assert len(received) == 10
    for frame_idx, masks in received:
        assert 1 in masks


def test_propagation_cancellation(runner, frame_320x180):
    stop = threading.Event()
    stop.set()
    frames = [(i, frame_320x180.copy()) for i in range(10)]
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    result = runner.propagate(_iter(frames), 0, 10, stop_event=stop)
    assert result.cancelled is True
    assert len(result.masks) == 0


def test_propagation_drift_moves_mask(runner, frame_320x180):
    """The mock drifts the ellipse centre 2px right per frame."""
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(80, 90, 1)]))
    frames = [(i, frame_320x180.copy()) for i in range(5)]
    result = runner.propagate(_iter(frames), 0, 5)

    # Mask centre-of-mass should shift right across frames
    def cx(mask: np.ndarray) -> float:
        ys, xs = np.nonzero(mask)
        return float(xs.mean()) if len(xs) else 0.0

    centres = [cx(result.masks[i][1]) for i in range(5)]
    assert centres[4] > centres[0]


def test_propagation_records_call(runner, frames_10):
    runner.set_prompts(1, ObjectPrompts(1, [PromptPoint(160, 90, 1)]))
    runner.propagate(_iter(frames_10), 0, 10)
    assert len(runner.propagation_calls) == 1
    assert runner.propagation_calls[0]["start_frame_idx"] == 0


def test_propagation_no_prompts_masks_empty(runner, frames_10):
    result = runner.propagate(_iter(frames_10), 0, 10)
    # Still returns per-frame entries but each is an empty dict
    for idx in range(10):
        assert result.masks[idx] == {}


# ── Confidence helpers ─────────────────────────────────────────────────────

@pytest.mark.parametrize("conf,expected", [
    (0.95, "high"),
    (0.75, "high"),
    (0.74, "medium"),
    (0.40, "medium"),
    (0.39, "low"),
    (0.01, "low"),
    (0.0,  "none"),
])
def test_confidence_level(conf, expected):
    assert confidence_level(conf) == expected


def test_mask_is_empty_true():
    assert mask_is_empty(np.zeros((180, 320), dtype=np.uint8))


def test_mask_is_empty_false():
    mask = np.zeros((180, 320), dtype=np.uint8)
    mask[50, 50] = 255
    assert not mask_is_empty(mask)


# ── create_runner factory ──────────────────────────────────────────────────

def test_create_runner_force_mock():
    r = create_runner(force_mock=True)
    assert isinstance(r, MockSAM3Runner)


def test_create_runner_no_sam2_falls_back():
    # sam2 is not installed on this machine — should always return mock
    r = create_runner(force_mock=False)
    assert isinstance(r, MockSAM3Runner)
