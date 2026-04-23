"""Tests for mask_exporter — export logic, manifest, feather, 16-bit output."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from sam3_resolve.core.mask_exporter import (
    ExportSettings,
    MaskExporter,
    _encode_frame,
)
from sam3_resolve.core.sam3_runner import PropagationResult


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_result():
    """1 object, 3 frames, 100×100 binary mask with centre filled."""
    r = PropagationResult(total_frames=3)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    for fi in range(3):
        r.masks[fi] = {1: mask.copy()}
    return r


@pytest.fixture()
def two_object_result():
    r = PropagationResult(total_frames=2)
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[10:50, 10:50] = 255
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[50:90, 50:90] = 255
    for fi in range(2):
        r.masks[fi] = {1: m1.copy(), 2: m2.copy()}
    return r


def make_settings(tmp_path, **kwargs) -> ExportSettings:
    return ExportSettings(
        output_dir=tmp_path,
        original_width=100,
        original_height=100,
        **kwargs,
    )


# ── _encode_frame (unit) ───────────────────────────────────────────────────

def test_encode_frame_creates_png(tmp_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    out = tmp_path / "obj_001" / "frame_000000.png"
    _encode_frame((1, 0, mask, str(out), 100, 100, 0))
    assert out.exists()


def test_encode_frame_is_16bit(tmp_path):
    mask = np.full((100, 100), 255, dtype=np.uint8)
    out = tmp_path / "frame.png"
    _encode_frame((1, 0, mask, str(out), 100, 100, 0))
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16


def test_encode_frame_white_maps_to_max(tmp_path):
    mask = np.full((50, 50), 255, dtype=np.uint8)
    out = tmp_path / "frame_white.png"
    _encode_frame((1, 0, mask, str(out), 50, 50, 0))
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert int(img.max()) == 65535


def test_encode_frame_black_maps_to_zero(tmp_path):
    mask = np.zeros((50, 50), dtype=np.uint8)
    out = tmp_path / "frame_black.png"
    _encode_frame((1, 0, mask, str(out), 50, 50, 0))
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert int(img.max()) == 0


def test_encode_frame_upscales(tmp_path):
    mask = np.full((50, 50), 255, dtype=np.uint8)
    out = tmp_path / "frame_up.png"
    _encode_frame((1, 0, mask, str(out), 100, 100, 0))  # orig 100×100
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert img.shape == (100, 100)


def test_encode_frame_feather_spreads_edge(tmp_path):
    """Feathering should produce non-255 pixels near the edge of the mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255          # tight centre square
    out = tmp_path / "frame_feather.png"
    _encode_frame((1, 0, mask, str(out), 100, 100, 10))  # large feather
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    # Pixels just outside the original mask should have intermediate values
    edge_val = int(img[44, 50])
    assert 0 < edge_val < 65535


# ── MaskExporter ──────────────────────────────────────────────────────────

def test_export_creates_object_folders(tmp_path, simple_result):
    s = make_settings(tmp_path)
    exp = MaskExporter(simple_result, s)
    result = exp.export()
    assert result.success
    assert (tmp_path / "object_001").is_dir()


def test_export_correct_frame_count(tmp_path, simple_result):
    s = make_settings(tmp_path)
    exp = MaskExporter(simple_result, s)
    result = exp.export()
    pngs = list((tmp_path / "object_001").glob("*.png"))
    assert len(pngs) == 3


def test_export_frames_written_count(tmp_path, simple_result):
    s = make_settings(tmp_path)
    result = MaskExporter(simple_result, s).export()
    assert result.frames_written == 3


def test_export_two_objects(tmp_path, two_object_result):
    s = make_settings(tmp_path)
    result = MaskExporter(two_object_result, s).export()
    assert result.success
    assert (tmp_path / "object_001").is_dir()
    assert (tmp_path / "object_002").is_dir()
    assert result.frames_written == 4   # 2 objects × 2 frames


def test_export_manifest_created(tmp_path, simple_result):
    s = make_settings(tmp_path)
    MaskExporter(simple_result, s).export()
    assert (tmp_path / "manifest.json").exists()


def test_export_manifest_content(tmp_path, simple_result):
    s = make_settings(tmp_path, object_names={1: "Hero"})
    MaskExporter(simple_result, s).export()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["objects"][0]["name"] == "Hero"
    assert manifest["objects"][0]["frame_count"] == 3
    assert manifest["objects"][0]["bit_depth"] == 16


def test_export_empty_result_writes_manifest(tmp_path):
    r = PropagationResult(total_frames=0)
    s = make_settings(tmp_path)
    result = MaskExporter(r, s).export()
    assert (tmp_path / "manifest.json").exists()
    assert result.frames_written == 0


def test_export_progress_callback(tmp_path, simple_result):
    calls = []
    def cb(done, total):
        calls.append((done, total))

    s = make_settings(tmp_path)
    MaskExporter(simple_result, s, progress_callback=cb).export()
    assert len(calls) == 3
    assert calls[-1] == (3, 3)


def test_export_cancel_stops_early(tmp_path, two_object_result):
    stop = threading.Event()
    stop.set()   # pre-cancelled

    s = make_settings(tmp_path)
    result = MaskExporter(two_object_result, s, stop_event=stop).export()
    assert result.cancelled


def test_export_result_output_dir(tmp_path, simple_result):
    s = make_settings(tmp_path)
    result = MaskExporter(simple_result, s).export()
    assert result.output_dir == tmp_path


def test_export_png_filenames(tmp_path, simple_result):
    s = make_settings(tmp_path)
    MaskExporter(simple_result, s).export()
    names = sorted(p.name for p in (tmp_path / "object_001").glob("*.png"))
    assert names[0] == "frame_000000.png"
    assert names[1] == "frame_000001.png"
    assert names[2] == "frame_000002.png"
