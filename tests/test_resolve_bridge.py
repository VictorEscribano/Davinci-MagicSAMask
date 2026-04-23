"""Unit tests for resolve_bridge — runs entirely via MockResolveBridge."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from sam3_resolve.core.resolve_bridge import (
    ClipFormat,
    ClipInfo,
    ClipOfflineError,
    FusionImportResult,
    MockResolveBridge,
    NoClipSelectedError,
    ResolveNotRunningError,
    UnsupportedClipError,
    _detect_format,
    create_bridge,
)


# ── _detect_format ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("/footage/clip.braw", ClipFormat.NEEDS_PROXY),
    ("/footage/clip.r3d",  ClipFormat.NEEDS_PROXY),
    ("/footage/clip.ari",  ClipFormat.NEEDS_PROXY),
    ("/footage/clip.mov",  ClipFormat.DIRECT),
    ("/footage/clip.mp4",  ClipFormat.DIRECT),
    ("/footage/clip.mkv",  ClipFormat.DIRECT),
    ("/footage/clip.mxf",  ClipFormat.DIRECT),
    ("/footage/clip.xyz",  ClipFormat.UNKNOWN),
])
def test_detect_format(path, expected):
    assert _detect_format(path) == expected


# ── ClipInfo helpers ───────────────────────────────────────────────────────

def test_clip_info_resolution_label():
    clip = MockResolveBridge.DEFAULT_CLIP
    assert clip.resolution_label == "1920×1080"


def test_clip_info_needs_proxy_false():
    clip = MockResolveBridge.DEFAULT_CLIP
    assert clip.needs_proxy is False


def test_clip_info_needs_proxy_true():
    clip = dataclasses.replace(
        MockResolveBridge.DEFAULT_CLIP,
        clip_format=ClipFormat.NEEDS_PROXY,
    )
    assert clip.needs_proxy is True


def test_clip_info_duration_seconds():
    clip = MockResolveBridge.DEFAULT_CLIP
    assert abs(clip.duration_seconds - 576 / 23.976) < 0.01


# ── MockResolveBridge normal flow ──────────────────────────────────────────

def test_mock_is_connected():
    bridge = MockResolveBridge()
    assert bridge.is_connected() is True


def test_mock_get_current_clip_returns_default():
    bridge = MockResolveBridge()
    clip = bridge.get_current_clip()
    assert clip.name == "A001C001_240315.mov"
    assert clip.fps == 23.976
    assert clip.clip_format == ClipFormat.DIRECT


def test_mock_get_selected_clips():
    bridge = MockResolveBridge()
    clips = bridge.get_selected_clips()
    assert len(clips) == 1
    assert clips[0] is bridge.mock_clip


def test_mock_custom_clip():
    custom = dataclasses.replace(
        MockResolveBridge.DEFAULT_CLIP,
        name="myfilm.mov",
        width=3840,
        height=2160,
    )
    bridge = MockResolveBridge(clip=custom)
    clip = bridge.get_current_clip()
    assert clip.name == "myfilm.mov"
    assert clip.resolution_label == "3840×2160"


# ── MockResolveBridge error simulation ────────────────────────────────────

def test_mock_simulate_no_clip_raises():
    bridge = MockResolveBridge()
    bridge.simulate_no_clip = True
    with pytest.raises(NoClipSelectedError):
        bridge.get_current_clip()


def test_mock_simulate_no_clip_returns_empty_list():
    bridge = MockResolveBridge()
    bridge.simulate_no_clip = True
    assert bridge.get_selected_clips() == []


def test_mock_simulate_offline_raises():
    bridge = MockResolveBridge()
    bridge.simulate_offline = True
    with pytest.raises(ClipOfflineError):
        bridge.get_current_clip()


# ── import_mask_to_fusion ──────────────────────────────────────────────────

def test_mock_import_mask_records_call(tmp_path):
    bridge = MockResolveBridge()
    clip = bridge.get_current_clip()
    result = bridge.import_mask_to_fusion(
        clip_info=clip,
        mask_folder=tmp_path / "masks",
        object_name="Person",
        object_index=1,
    )
    assert result.success is True
    assert "SAM3_Mask_Obj1_Loader" in result.node_names
    assert "SAM3_Mask_Obj1_Matte" in result.node_names
    assert len(bridge.imported_masks) == 1
    assert bridge.imported_masks[0]["object_index"] == 1


def test_mock_import_mask_multiple_objects(tmp_path):
    bridge = MockResolveBridge()
    clip = bridge.get_current_clip()
    for i in range(1, 4):
        bridge.import_mask_to_fusion(clip, tmp_path, f"Object {i}", i)
    assert len(bridge.imported_masks) == 3
    labels = [r["node_label"] for r in bridge.imported_masks]
    assert labels == ["SAM3_Mask_Obj1", "SAM3_Mask_Obj2", "SAM3_Mask_Obj3"]


# ── render_proxy_via_resolve ───────────────────────────────────────────────

def test_mock_render_proxy_returns_true(tmp_path):
    bridge = MockResolveBridge()
    clip = bridge.get_current_clip()
    result = bridge.render_proxy_via_resolve(clip, 0.25, tmp_path / "proxy.mp4")
    assert result is True


# ── create_bridge factory ──────────────────────────────────────────────────

def test_create_bridge_force_mock_returns_mock():
    bridge = create_bridge(force_mock=True)
    assert isinstance(bridge, MockResolveBridge)


def test_create_bridge_auto_falls_back_to_mock():
    # No Resolve installed on this machine → should always fall back
    bridge = create_bridge(force_mock=False)
    assert isinstance(bridge, MockResolveBridge)


# ── FusionImportResult ────────────────────────────────────────────────────

def test_fusion_import_result_defaults():
    result = FusionImportResult(success=True, clip_name="test.mov")
    assert result.node_names == []
    assert result.error == ""
