"""Tests for media_handler — timecode, scaling, VFR, scene cuts, frame I/O."""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sam3_resolve.core.media_handler import (
    MediaHandler,
    ProxyInfo,
    ProxyPreset,
    ScaleFactor,
    detect_vfr,
    frame_to_timecode,
    timecode_to_frame,
)
from sam3_resolve.core.resolve_bridge import ClipFormat, ClipInfo, MockResolveBridge


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def clip_1080p() -> ClipInfo:
    return MockResolveBridge.DEFAULT_CLIP


@pytest.fixture()
def clip_4k() -> ClipInfo:
    return dataclasses.replace(
        MockResolveBridge.DEFAULT_CLIP,
        width=3840,
        height=2160,
        fps=24.0,
        duration_frames=240,
        in_point_frame=0,
        out_point_frame=240,
        media_pool_uuid="mock-uuid-4k",
    )


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    """
    Write a tiny 10-frame CFR MP4 where every frame has natural-looking noise
    so consecutive histogram distances stay low (no spurious scene cuts).
    """
    out = tmp_path / "test.mp4"
    rng = np.random.default_rng(42)
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (320, 180),
    )
    for _ in range(10):
        # Gaussian noise around 128 — each frame has similar histogram
        frame = np.clip(rng.normal(128, 15, (180, 320, 3)), 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return out


@pytest.fixture()
def scene_cut_video(tmp_path: Path) -> Path:
    """10 dark frames then 10 bright frames — one hard cut."""
    out = tmp_path / "scene_cut.mp4"
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (320, 180),
    )
    for _ in range(10):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    for _ in range(10):
        writer.write(np.full((180, 320, 3), 220, dtype=np.uint8))
    writer.release()
    return out


# ── timecode_to_frame ──────────────────────────────────────────────────────

@pytest.mark.parametrize("tc,fps,expected", [
    ("00:00:00:00", 24.0,   0),
    ("00:00:01:00", 24.0,  24),
    ("00:00:01:12", 24.0,  36),
    ("00:01:00:00", 24.0, 1440),
    ("01:00:00:00", 24.0, 86400),
    ("00:00:00:00", 25.0,   0),
    ("00:00:01:00", 25.0,  25),
    ("00:00:00:00", 30.0,   0),
    ("00:00:01:00", 30.0,  30),
])
def test_timecode_to_frame_ndf(tc, fps, expected):
    assert timecode_to_frame(tc, fps) == expected


def test_timecode_to_frame_drop_frame_semicolon():
    # At 00:01:00;02, drop-frame skips frames :00 and :01 at the 1-minute mark.
    # SMPTE formula: 30*60 + 2 - 2*(1 - 0) = 1800. The semicolon activates
    # drop-frame mode automatically.
    result_df = timecode_to_frame("00:01:00;02", 29.97)
    assert result_df == 1800
    # Non-drop-frame for same string (no semicolon) gives a different count
    result_ndf = timecode_to_frame("00:01:00:02", 29.97, drop_frame=False)
    assert result_ndf == 1802


def test_timecode_to_frame_invalid_raises():
    with pytest.raises(ValueError):
        timecode_to_frame("not:a:timecode", 24.0)

    with pytest.raises(ValueError):
        timecode_to_frame("00:00:00", 24.0)  # only 3 parts


def test_timecode_roundtrip():
    for frame in [0, 1, 23, 24, 100, 1440, 86399]:
        tc = frame_to_timecode(frame, 24.0)
        assert timecode_to_frame(tc, 24.0) == frame


# ── frame_to_timecode ──────────────────────────────────────────────────────

def test_frame_to_timecode_basic():
    assert frame_to_timecode(0, 24.0) == "00:00:00:00"
    assert frame_to_timecode(24, 24.0) == "00:00:01:00"
    assert frame_to_timecode(25, 24.0) == "00:00:01:01"
    assert frame_to_timecode(1440, 24.0) == "00:01:00:00"


# ── ProxyPreset ────────────────────────────────────────────────────────────

def test_proxy_preset_scales():
    assert ProxyPreset.scale(ProxyPreset.QUARTER) == 0.25
    assert ProxyPreset.scale(ProxyPreset.HALF) == 0.5
    assert ProxyPreset.scale(ProxyPreset.FULL) == 1.0


def test_proxy_preset_unknown_defaults_quarter():
    assert ProxyPreset.scale("unknown") == 0.25


# ── ScaleFactor / compute_scale_factors ───────────────────────────────────

def test_compute_scale_factors_quarter(clip_1080p):
    handler = MediaHandler(clip_1080p)
    proxy = ProxyInfo(
        path=Path("/fake/proxy.mp4"),
        width=480, height=270,
        scale_x=0.25, scale_y=0.25,
        source="generated",
    )
    sf = handler.compute_scale_factors(proxy)
    assert sf.x == pytest.approx(4.0)
    assert sf.y == pytest.approx(4.0)
    assert sf.original_width == 1920
    assert sf.proxy_width == 480


def test_compute_scale_factors_half(clip_4k):
    handler = MediaHandler(clip_4k)
    proxy = ProxyInfo(
        path=Path("/fake/proxy.mp4"),
        width=1920, height=1080,
        scale_x=0.5, scale_y=0.5,
        source="generated",
    )
    sf = handler.compute_scale_factors(proxy)
    assert sf.x == pytest.approx(2.0)
    assert sf.y == pytest.approx(2.0)


# ── upscale_mask ───────────────────────────────────────────────────────────

def test_upscale_mask_correct_output_size(clip_1080p):
    handler = MediaHandler(clip_1080p)
    proxy = ProxyInfo(Path("/x"), 480, 270, 0.25, 0.25, "generated")
    sf = handler.compute_scale_factors(proxy)

    small_mask = np.zeros((270, 480), dtype=np.uint8)
    small_mask[100:170, 150:330] = 255  # white rectangle

    upscaled = handler.upscale_mask(small_mask, sf)
    assert upscaled.shape == (1080, 1920)


def test_upscale_mask_has_content(clip_1080p):
    handler = MediaHandler(clip_1080p)
    proxy = ProxyInfo(Path("/x"), 480, 270, 0.25, 0.25, "generated")
    sf = handler.compute_scale_factors(proxy)

    mask = np.zeros((270, 480), dtype=np.uint8)
    mask[50:220, 100:380] = 255

    upscaled = handler.upscale_mask(mask, sf)
    assert upscaled.max() > 0


def test_upscale_mask_feather_softens_edges(clip_1080p):
    """Feathering expands the non-zero region — the contour bleeds outward."""
    handler = MediaHandler(clip_1080p)
    proxy = ProxyInfo(Path("/x"), 480, 270, 0.25, 0.25, "generated")
    sf = handler.compute_scale_factors(proxy)

    mask = np.zeros((270, 480), dtype=np.uint8)
    mask[135, :] = 255   # single horizontal line

    no_feather = handler.upscale_mask(mask, sf, feather_px=0)
    with_feather = handler.upscale_mask(mask, sf, feather_px=20)

    # Larger sigma → more non-zero pixels (contour bleeds further out)
    nonzero_before = int((no_feather > 0).sum())
    nonzero_after = int((with_feather > 0).sum())
    assert nonzero_after > nonzero_before


# ── find_existing_proxy ────────────────────────────────────────────────────

def test_find_existing_proxy_resolve_path(clip_1080p, tmp_path):
    proxy_file = tmp_path / "proxy.mp4"
    # Create a minimal video that OpenCV can open
    writer = cv2.VideoWriter(
        str(proxy_file), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (480, 270)
    )
    writer.write(np.zeros((270, 480, 3), dtype=np.uint8))
    writer.release()

    clip = dataclasses.replace(clip_1080p, proxy_path=str(proxy_file))
    handler = MediaHandler(clip)
    info = handler.find_existing_proxy()

    assert info is not None
    assert info.source == "resolve"
    assert info.width == 480


def test_find_existing_proxy_sidecar(clip_1080p, tmp_path):
    original = tmp_path / "A001C001_240315.mov"
    original.touch()
    sidecar = tmp_path / "A001C001_240315_proxy.mov"

    writer = cv2.VideoWriter(
        str(sidecar), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (960, 540)
    )
    writer.write(np.zeros((540, 960, 3), dtype=np.uint8))
    writer.release()

    clip = dataclasses.replace(clip_1080p, file_path=str(original))
    handler = MediaHandler(clip)
    info = handler.find_existing_proxy()

    assert info is not None
    assert info.source == "sidecar"
    assert info.width == 960


def test_find_existing_proxy_returns_none_when_missing(clip_1080p, tmp_path):
    clip = dataclasses.replace(
        clip_1080p,
        file_path=str(tmp_path / "nonexistent.mov"),
        proxy_path="",
    )
    handler = MediaHandler(clip, proxy_folder_override=tmp_path)
    assert handler.find_existing_proxy() is None


# ── MediaHandler frame read ────────────────────────────────────────────────

def test_read_frame_returns_array(clip_1080p, synthetic_video, tmp_path):
    # Point clip at synthetic video
    clip = dataclasses.replace(
        clip_1080p,
        file_path=str(synthetic_video),
        width=320, height=180,
        duration_frames=10,
        in_point_frame=0, out_point_frame=10,
        media_pool_uuid="test-read-frame",
    )
    handler = MediaHandler(clip)
    handler.open_video()
    frame = handler.read_frame(0, use_cache=False)
    handler.close_video()

    assert frame is not None
    assert frame.shape == (180, 320, 3)


def test_iter_frames_yields_all(clip_1080p, synthetic_video):
    clip = dataclasses.replace(
        clip_1080p,
        file_path=str(synthetic_video),
        width=320, height=180,
        duration_frames=10,
        in_point_frame=0, out_point_frame=10,
        media_pool_uuid="test-iter-frames",
    )
    handler = MediaHandler(clip)
    handler.open_video()
    frames = list(handler.iter_frames(use_cache=False))
    handler.close_video()

    assert len(frames) == 10
    indices, _ = zip(*frames)
    assert list(indices) == list(range(10))


def test_iter_frames_stop_event(clip_1080p, synthetic_video):
    clip = dataclasses.replace(
        clip_1080p,
        file_path=str(synthetic_video),
        width=320, height=180,
        duration_frames=10,
        in_point_frame=0, out_point_frame=10,
        media_pool_uuid="test-stop-event",
    )
    stop = threading.Event()
    stop.set()   # already stopped

    handler = MediaHandler(clip)
    handler.open_video()
    frames = list(handler.iter_frames(use_cache=False, stop_event=stop))
    handler.close_video()

    assert frames == []


def test_context_manager_closes_video(clip_1080p, synthetic_video):
    clip = dataclasses.replace(
        clip_1080p,
        file_path=str(synthetic_video),
        media_pool_uuid="test-ctx-mgr",
    )
    with MediaHandler(clip) as handler:
        handler.open_video()
        assert handler._cap is not None
    assert handler._cap is None


# ── scene cut detection ────────────────────────────────────────────────────

def test_detect_scene_cuts_finds_cut(scene_cut_video):
    from sam3_resolve.core.media_handler import detect_scene_cuts
    cuts = detect_scene_cuts(scene_cut_video, 0, 20, threshold=0.3)
    # Expect at least one cut between frame 9 and 10
    assert len(cuts) >= 1
    assert any(8 <= c <= 12 for c in cuts)


def test_detect_scene_cuts_uniform_no_cut(synthetic_video):
    from sam3_resolve.core.media_handler import detect_scene_cuts
    # synthetic_video has gradually changing brightness — no hard cuts
    cuts = detect_scene_cuts(synthetic_video, 0, 10, threshold=0.3)
    assert len(cuts) == 0


def test_detect_scene_cuts_stop_event(scene_cut_video):
    from sam3_resolve.core.media_handler import detect_scene_cuts
    stop = threading.Event()
    stop.set()
    cuts = detect_scene_cuts(scene_cut_video, 0, 20, stop_event=stop)
    assert cuts == []


# ── estimate_proxy_size_mb ────────────────────────────────────────────────

def test_estimate_proxy_size_positive():
    size = MediaHandler.estimate_proxy_size_mb(480, 270, 576, 24.0)
    assert size > 0


def test_estimate_proxy_size_scales_with_resolution():
    size_hd = MediaHandler.estimate_proxy_size_mb(1920, 1080, 240, 24.0)
    size_qhd = MediaHandler.estimate_proxy_size_mb(480, 270, 240, 24.0)
    assert size_hd > size_qhd
