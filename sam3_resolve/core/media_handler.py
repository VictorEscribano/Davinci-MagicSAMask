"""
Media handler: frame extraction, proxy detection/generation, VFR handling,
timecode conversion, and scene-cut detection.

All heavy operations (proxy generation, full frame extraction) are designed
to be called from a QThread worker; they accept an optional stop_event and
emit progress via a callback rather than blocking the caller.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from sam3_resolve.constants import (
    CACHE_DIR,
    FRAME_JPEG_QUALITY,
    MASK_UPSCALE_BLUR_SIGMA,
    MASK_UPSCALE_MAX_CROP_PX,
    PROXY_CRF,
    PROXY_PRESET,
    PROXY_SCALE_FULL,
    PROXY_SCALE_HALF,
    PROXY_SCALE_QUARTER,
    SCENE_CUT_HIST_DIFF_THRESHOLD,
)
from sam3_resolve.core.resolve_bridge import ClipInfo

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, float], None]  # (current, total, fps)


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProxyInfo:
    """Describes a found or generated proxy file."""
    path: Path
    width: int
    height: int
    scale_x: float
    scale_y: float
    source: str   # 'resolve' | 'sidecar' | 'config_folder' | 'generated'

    @property
    def resolution_label(self) -> str:
        return f"{self.width}×{self.height}"


@dataclass(frozen=True)
class ScaleFactor:
    """Proxy-to-original scale factors used for mask upscaling."""
    x: float
    y: float
    proxy_width: int
    proxy_height: int
    original_width: int
    original_height: int


class ProxyPreset:
    QUARTER = "quarter"
    HALF = "half"
    FULL = "full"

    @staticmethod
    def scale(preset: str) -> float:
        mapping = {
            ProxyPreset.QUARTER: PROXY_SCALE_QUARTER,
            ProxyPreset.HALF: PROXY_SCALE_HALF,
            ProxyPreset.FULL: PROXY_SCALE_FULL,
        }
        return mapping.get(preset, PROXY_SCALE_QUARTER)

    @staticmethod
    def label(preset: str) -> str:
        return {
            ProxyPreset.QUARTER: "1/4 res",
            ProxyPreset.HALF: "1/2 res",
            ProxyPreset.FULL: "Full resolution",
        }.get(preset, "1/4 res")


# ── Timecode utilities ─────────────────────────────────────────────────────

def timecode_to_frame(timecode: str, fps: float, drop_frame: bool = False) -> int:
    """
    Convert an HH:MM:SS:FF timecode string to an absolute frame index.

    Args:
        timecode:   String in the form "HH:MM:SS:FF" or "HH:MM:SS;FF"
                    (semicolon indicates drop-frame).
        fps:        Clip frame rate (e.g. 23.976, 29.97, 24.0).
        drop_frame: If True, apply drop-frame correction for 29.97/59.94 fps.
                    Detected automatically from a semicolon separator.

    Returns:
        Zero-based frame index.

    Raises:
        ValueError: If the timecode string cannot be parsed.
    """
    tc = timecode.strip()

    # Detect drop-frame from semicolon separator
    if ";" in tc:
        drop_frame = True
        tc = tc.replace(";", ":")

    parts = tc.split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: '{timecode}'")

    try:
        hh, mm, ss, ff = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Non-integer timecode component in '{timecode}'") from exc

    nominal_fps = round(fps)  # 24, 25, 30 …

    if drop_frame and nominal_fps in (30, 60):
        # SMPTE drop-frame algorithm
        drop_frames = 2 if nominal_fps == 30 else 4
        total_minutes = 60 * hh + mm
        frame_number = (
            nominal_fps * 3600 * hh
            + nominal_fps * 60 * mm
            + nominal_fps * ss
            + ff
            - drop_frames * (total_minutes - total_minutes // 10)
        )
    else:
        frame_number = nominal_fps * (3600 * hh + 60 * mm + ss) + ff

    return frame_number


def frame_to_timecode(frame: int, fps: float) -> str:
    """
    Convert a zero-based frame index back to a non-drop-frame timecode string.

    Args:
        frame: Frame index.
        fps:   Clip frame rate.

    Returns:
        String "HH:MM:SS:FF".
    """
    nominal_fps = max(1, round(fps))
    ff = frame % nominal_fps
    total_sec = frame // nominal_fps
    ss = total_sec % 60
    total_min = total_sec // 60
    mm = total_min % 60
    hh = total_min // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ── VFR detection ──────────────────────────────────────────────────────────

def detect_vfr(video_path: Path) -> bool:
    """
    Return True if the container likely has variable frame rate.

    Uses ffprobe to read the video stream's r_frame_rate vs avg_frame_rate.
    A significant difference between the two indicates VFR.

    Args:
        video_path: Path to the video file.

    Returns:
        True if VFR is detected, False for CFR or if ffprobe is unavailable.
    """
    if not shutil.which("ffprobe"):
        logger.warning("ffprobe not found; skipping VFR check")
        return False

    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return False

        stream = streams[0]

        def _parse_rate(rate_str: str) -> float:
            if "/" in rate_str:
                num, den = rate_str.split("/")
                return float(num) / float(den) if float(den) else 0.0
            return float(rate_str)

        r_fps = _parse_rate(stream.get("r_frame_rate", "0/1"))
        avg_fps = _parse_rate(stream.get("avg_frame_rate", "0/1"))

        if avg_fps == 0:
            return False

        diff_ratio = abs(r_fps - avg_fps) / avg_fps
        is_vfr = diff_ratio > 0.01   # >1% difference
        if is_vfr:
            logger.warning(
                "VFR detected in %s: r_fps=%.4f avg_fps=%.4f",
                video_path.name, r_fps, avg_fps,
            )
        return is_vfr
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("VFR detection failed for %s: %s", video_path.name, exc)
        return False


# ── Scene-cut detection ────────────────────────────────────────────────────

def detect_scene_cuts(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    threshold: float = SCENE_CUT_HIST_DIFF_THRESHOLD,
    stop_event: Optional[threading.Event] = None,
) -> list[int]:
    """
    Return frame indices where a hard scene cut occurs.

    Uses L1 distance between normalised luminance histograms of consecutive
    frames. O(256) per frame — fast enough to run during proxy generation.

    Args:
        video_path:   Video file to analyse.
        start_frame:  First frame to examine (inclusive).
        end_frame:    Last frame to examine (exclusive).
        threshold:    L1 distance above which a cut is declared.
        stop_event:   Optional threading.Event; if set, stops early.

    Returns:
        Sorted list of frame indices immediately after detected cuts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open %s for scene-cut detection", video_path)
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cuts: list[int] = []

    prev_hist: Optional[np.ndarray] = None
    frame_idx = start_frame

    try:
        while frame_idx < end_frame:
            if stop_event and stop_event.is_set():
                break

            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if diff > threshold:
                    cuts.append(frame_idx)
                    logger.debug("Scene cut at frame %d (diff=%.3f)", frame_idx, diff)

            prev_hist = hist
            frame_idx += 1
    finally:
        cap.release()

    return cuts


# ── Frame cache ────────────────────────────────────────────────────────────

def _frame_cache_dir(clip_uuid: str) -> Path:
    d = CACHE_DIR / clip_uuid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _frame_cache_path(clip_uuid: str, frame_idx: int) -> Path:
    return _frame_cache_dir(clip_uuid) / f"frame_{frame_idx:06d}.jpg"


# ── Core media handler ─────────────────────────────────────────────────────

class MediaHandler:
    """
    Manages all media I/O for a single clip session.

    Responsibilities:
      - Proxy detection and generation
      - Frame extraction (lazy, JPEG-cached)
      - VFR and scene-cut detection
      - Resolution scale factor computation for mask upscaling
    """

    def __init__(
        self,
        clip_info: ClipInfo,
        proxy_folder_override: Optional[Path] = None,
    ) -> None:
        """
        Args:
            clip_info:             Metadata for the clip being processed.
            proxy_folder_override: User-defined proxy folder from config.
                                   If None, uses CACHE_DIR.
        """
        self.clip = clip_info
        self._proxy_folder_override = proxy_folder_override
        self._proxy: Optional[ProxyInfo] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._active_video_path: Optional[Path] = None

    # ── Proxy detection ────────────────────────────────────────────────────

    def find_existing_proxy(self) -> Optional[ProxyInfo]:
        """
        Search for an existing proxy in priority order:
          1. Resolve-assigned proxy path (from ClipInfo.proxy_path)
          2. Sidecar files beside the original (_proxy / _LR suffix)
          3. User-defined proxy folder (proxy_folder_override)

        Returns:
            ProxyInfo if found, None otherwise.
        """
        # 1 — Resolve proxy
        if self.clip.proxy_path:
            p = Path(self.clip.proxy_path)
            if p.exists():
                info = self._probe_proxy(p, source="resolve")
                if info:
                    logger.info("Using Resolve proxy: %s", p)
                    return info

        # 2 — Sidecar
        original = Path(self.clip.file_path)
        for suffix in ("_proxy", "_LR", "_proxy_LR"):
            candidate = original.with_name(original.stem + suffix + original.suffix)
            if candidate.exists():
                info = self._probe_proxy(candidate, source="sidecar")
                if info:
                    logger.info("Using sidecar proxy: %s", candidate)
                    return info
            # Also check common proxy container extensions
            for ext in (".mp4", ".mov"):
                candidate2 = original.with_name(original.stem + suffix).with_suffix(ext)
                if candidate2.exists():
                    info = self._probe_proxy(candidate2, source="sidecar")
                    if info:
                        return info

        # 3 — User-defined folder
        if self._proxy_folder_override:
            for ext in (".mp4", ".mov", ".mkv"):
                candidate3 = self._proxy_folder_override / (original.stem + ext)
                if candidate3.exists():
                    info = self._probe_proxy(candidate3, source="config_folder")
                    if info:
                        logger.info("Using config-folder proxy: %s", candidate3)
                        return info

        return None

    def _probe_proxy(self, path: Path, source: str) -> Optional["ProxyInfo"]:
        """Read resolution of a proxy file and compute scale factors."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == 0 or h == 0:
            return None
        return ProxyInfo(
            path=path,
            width=w,
            height=h,
            scale_x=w / self.clip.width,
            scale_y=h / self.clip.height,
            source=source,
        )

    # ── Proxy generation ───────────────────────────────────────────────────

    def generate_proxy(
        self,
        preset: str = ProxyPreset.QUARTER,
        output_path: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> ProxyInfo:
        """
        Generate a proxy file using ffmpeg.

        Args:
            preset:            One of ProxyPreset constants.
            output_path:       Destination file. If None, auto-generates in
                               CACHE_DIR / <clip_uuid> / proxy.mp4.
            progress_callback: Called with (current_frame, total_frames, fps).
            stop_event:        Set to abort mid-generation.

        Returns:
            ProxyInfo for the generated file.

        Raises:
            RuntimeError: If ffmpeg is not found or exits with an error.
            FileNotFoundError: If the source clip file does not exist.
        """
        source = Path(self.clip.file_path)
        if not source.exists():
            raise FileNotFoundError(
                f"Source clip not found: {source}. Relink the clip in the Media Pool."
            )

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found. Install it or set the path in Settings → ffmpeg binary."
            )

        if output_path is None:
            output_path = (
                CACHE_DIR / self.clip.media_pool_uuid / f"proxy_{preset}.mp4"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scale = ProxyPreset.scale(preset)
        target_w = int(self.clip.width * scale)
        target_h = int(self.clip.height * scale)
        # Ensure dimensions are even (required by libx264)
        target_w += target_w % 2
        target_h += target_h % 2

        cmd = [
            "ffmpeg", "-y",
            "-i", str(source),
            "-vf", f"scale={target_w}:{target_h}",
            "-c:v", "libx264",
            "-crf", str(PROXY_CRF),
            "-preset", PROXY_PRESET,
            "-an",
            "-progress", "pipe:2",
            str(output_path),
        ]

        logger.info(
            "Generating %s proxy: %s → %s",
            ProxyPreset.label(preset), source.name, output_path,
        )

        self._run_ffmpeg_with_progress(
            cmd,
            total_frames=self.clip.out_point_frame - self.clip.in_point_frame,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )

        if not output_path.exists():
            raise RuntimeError(f"ffmpeg did not produce output at {output_path}")

        info = self._probe_proxy(output_path, source="generated")
        if info is None:
            raise RuntimeError(f"Generated proxy at {output_path} could not be read")

        logger.info(
            "Proxy generated: %s (%d×%d)", output_path.name, info.width, info.height
        )
        return info

    @staticmethod
    def _run_ffmpeg_with_progress(
        cmd: list[str],
        total_frames: int,
        progress_callback: Optional[ProgressCallback],
        stop_event: Optional[threading.Event],
    ) -> None:
        """
        Run an ffmpeg command and parse its `-progress pipe:2` output.

        Args:
            cmd:               Full ffmpeg command list.
            total_frames:      Expected frame count (for % calculation).
            progress_callback: Called each time a new frame count is parsed.
            stop_event:        Set to kill ffmpeg early.

        Raises:
            RuntimeError: If ffmpeg exits with non-zero status.
        """
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        current_frame = 0
        current_fps = 0.0
        kv: dict[str, str] = {}

        assert process.stderr is not None
        for line in process.stderr:
            if stop_event and stop_event.is_set():
                process.kill()
                logger.info("ffmpeg proxy generation cancelled")
                return

            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                kv[key.strip()] = val.strip()

                if key.strip() == "progress":
                    frame_str = kv.get("frame", "0")
                    fps_str = kv.get("fps", "0")
                    try:
                        current_frame = int(frame_str)
                        current_fps = float(fps_str)
                    except ValueError:
                        pass
                    if progress_callback and total_frames > 0:
                        progress_callback(current_frame, total_frames, current_fps)
                    kv.clear()

        process.wait()
        if process.returncode not in (0, None) and not (
            stop_event and stop_event.is_set()
        ):
            raise RuntimeError(
                f"ffmpeg exited with code {process.returncode}. "
                "Check the log for details."
            )

    # ── Frame extraction ───────────────────────────────────────────────────

    def open_video(self, use_proxy: Optional[ProxyInfo] = None) -> None:
        """
        Open the video for frame-by-frame reading.

        Args:
            use_proxy: If provided, opens the proxy file instead of the original.
        """
        if self._cap is not None:
            self._cap.release()

        video_path = use_proxy.path if use_proxy else Path(self.clip.file_path)
        self._active_video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Cannot open video: {video_path}. "
                "The file may be offline, corrupted, or an unsupported format."
            )

        logger.debug("Opened video: %s", video_path.name)

    def close_video(self) -> None:
        """Release the OpenCV VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(
        self,
        frame_index: int,
        use_cache: bool = True,
        proxy: Optional[ProxyInfo] = None,
    ) -> np.ndarray:
        """
        Read a single frame, optionally from the JPEG cache.

        Frame index is relative to the source file (not the in/out range).

        Args:
            frame_index: Absolute frame index in the source (or proxy) file.
            use_cache:   If True, check disk cache first; write on miss.
            proxy:       If provided, map frame_index through the proxy's
                         frame count (proxy may differ in duration at same fps).

        Returns:
            BGR numpy array (H, W, 3).

        Raises:
            RuntimeError: If the video is not open or the frame cannot be read.
        """
        cache_path = _frame_cache_path(self.clip.media_pool_uuid, frame_index)

        if use_cache and cache_path.exists():
            img = cv2.imread(str(cache_path))
            if img is not None:
                return img

        if self._cap is None:
            raise RuntimeError("Video is not open. Call open_video() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} from "
                f"{self._active_video_path}."
            )

        if use_cache:
            cv2.imwrite(
                str(cache_path), frame,
                [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY],
            )

        return frame

    def iter_frames(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        use_cache: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yield (frame_index, frame_array) for the clip's in/out range.

        Args:
            start:      Override start frame (default: clip.in_point_frame).
            end:        Override end frame exclusive (default: clip.out_point_frame).
            use_cache:  Use/populate the JPEG cache.
            stop_event: Set to abort mid-iteration.

        Yields:
            (frame_index, BGR numpy array) tuples.
        """
        first = start if start is not None else self.clip.in_point_frame
        last = end if end is not None else self.clip.out_point_frame

        if self._cap is None:
            raise RuntimeError("Video is not open. Call open_video() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for idx in range(first, last):
            if stop_event and stop_event.is_set():
                logger.info("Frame iteration cancelled at frame %d", idx)
                return

            cache_path = _frame_cache_path(self.clip.media_pool_uuid, idx)
            if use_cache and cache_path.exists():
                img = cv2.imread(str(cache_path))
                if img is not None:
                    yield idx, img
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.warning("Frame %d unreadable; stopping iteration", idx)
                return

            if use_cache:
                cv2.imwrite(
                    str(cache_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY],
                )

            yield idx, frame

    # ── Scale factors ──────────────────────────────────────────────────────

    def compute_scale_factors(self, proxy: ProxyInfo) -> ScaleFactor:
        """
        Compute proxy→original scale factors for mask upscaling.

        Args:
            proxy: The proxy being used for inference.

        Returns:
            ScaleFactor with x/y ratios and both resolutions.
        """
        return ScaleFactor(
            x=self.clip.width / proxy.width,
            y=self.clip.height / proxy.height,
            proxy_width=proxy.width,
            proxy_height=proxy.height,
            original_width=self.clip.width,
            original_height=self.clip.height,
        )

    def upscale_mask(
        self,
        mask: np.ndarray,
        scale: ScaleFactor,
        feather_px: float = 0.0,
    ) -> np.ndarray:
        """
        Upscale a proxy-resolution binary mask to the original clip resolution.

        Args:
            mask:       Single-channel uint8 mask (0 or 255).
            scale:      ScaleFactor from compute_scale_factors().
            feather_px: Gaussian blur radius (σ) to soften edges after upscale.
                        Pass 0 to skip.

        Returns:
            Upscaled single-channel uint8 mask matching original resolution,
            padded or cropped by at most MASK_UPSCALE_MAX_CROP_PX if needed.
        """
        target_w = scale.original_width
        target_h = scale.original_height

        upscaled = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Fix rounding differences from ffmpeg (≤ MASK_UPSCALE_MAX_CROP_PX)
        h, w = upscaled.shape[:2]
        if (w, h) != (target_w, target_h):
            delta_w = abs(w - target_w)
            delta_h = abs(h - target_h)
            if delta_w <= MASK_UPSCALE_MAX_CROP_PX and delta_h <= MASK_UPSCALE_MAX_CROP_PX:
                upscaled = cv2.resize(upscaled, (target_w, target_h))
            else:
                logger.warning(
                    "Mask size mismatch too large to auto-correct: "
                    "got %dx%d, expected %dx%d",
                    w, h, target_w, target_h,
                )

        # Artifact softening (always applied after upscale)
        if MASK_UPSCALE_BLUR_SIGMA > 0:
            ksize = 0  # cv2 auto-sizes kernel from sigma
            upscaled = cv2.GaussianBlur(
                upscaled, (ksize, ksize), MASK_UPSCALE_BLUR_SIGMA
            )

        # Optional feathering
        if feather_px > 0:
            sigma = feather_px / 3.0
            upscaled = cv2.GaussianBlur(upscaled, (0, 0), sigma)

        return upscaled

    # ── Cache management ───────────────────────────────────────────────────

    def clear_frame_cache(self) -> int:
        """
        Delete all cached frames for this clip.

        Returns:
            Number of files deleted.
        """
        cache_dir = _frame_cache_dir(self.clip.media_pool_uuid)
        count = 0
        for f in cache_dir.glob("frame_*.jpg"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cleared %d cached frames for clip %s", count, self.clip.name)
        return count

    @staticmethod
    def cache_size_bytes(clip_uuid: str) -> int:
        """Return total disk usage of the frame cache for one clip in bytes."""
        cache_dir = CACHE_DIR / clip_uuid
        if not cache_dir.exists():
            return 0
        return sum(f.stat().st_size for f in cache_dir.glob("frame_*.jpg"))

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "MediaHandler":
        return self

    def __exit__(self, *_) -> None:
        self.close_video()

    # ── Estimated processing time ──────────────────────────────────────────

    @staticmethod
    def estimate_proxy_size_mb(
        width: int, height: int, duration_frames: int, fps: float
    ) -> float:
        """
        Rough estimate of proxy file size in MB for UI display.

        Uses empirical ~0.5 Mbps per 360×202 equivalent (CRF 18, libx264 fast).

        Args:
            width, height:    Target proxy resolution.
            duration_frames:  Number of frames.
            fps:              Frame rate.

        Returns:
            Estimated size in megabytes.
        """
        pixels = width * height
        reference_pixels = 360 * 202
        reference_bitrate_mbps = 0.5
        bitrate_mbps = reference_bitrate_mbps * (pixels / reference_pixels)
        duration_s = duration_frames / fps if fps else 0.0
        return bitrate_mbps * duration_s / 8.0
