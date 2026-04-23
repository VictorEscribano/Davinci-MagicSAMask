"""
SAM3 inference runner.

Manages model loading, prompt state for up to 8 objects, live single-frame
inference, and full-video propagation.  Pure Python — no PyQt6 dependency.
QThread workers that call this module live in ui/workers.py.

Mock mode (MockSAM3Runner) is activated automatically when SAM3 / torch are
not installed, enabling full UI development without a GPU.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional

import numpy as np

from sam3_resolve.constants import (
    MAX_OBJECTS,
    MODELS_DIR,
    OOM_RETRY_BATCH_SIZE,
    SAM3_BASE_FILENAME,
    SAM3_LARGE_FILENAME,
    SAM3_LOGIT_THRESHOLD,
    VRAM_THRESHOLD_LARGE_GB,
)
from sam3_resolve.core.gpu_utils import Backend, GPUInfo, recommended_dtype, torch_device_string

logger = logging.getLogger(__name__)

# Callback types
ProgressCallback = Callable[[int, int, float, float], None]   # frame, total, fps, eta_s
MaskCallback = Callable[[int, dict[int, np.ndarray]], None]    # frame_idx, {obj_id: mask}


# ── Prompt data types ──────────────────────────────────────────────────────

@dataclass
class PromptPoint:
    """A single click prompt."""
    x: float
    y: float
    label: int   # 1 = positive (include), 0 = negative (exclude)

    def as_numpy(self) -> np.ndarray:
        return np.array([[self.x, self.y]], dtype=np.float32)


@dataclass
class PromptBox:
    """A bounding box prompt."""
    x1: float
    y1: float
    x2: float
    y2: float

    def as_numpy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class ObjectPrompts:
    """All prompts for one tracked object."""
    object_id: int
    points: list[PromptPoint] = field(default_factory=list)
    box: Optional[PromptBox] = None
    mask_prompt: Optional[np.ndarray] = None   # external mask as input prompt
    frame_idx: int = 0                          # video frame where prompts were placed

    def has_prompts(self) -> bool:
        positive = any(p.label == 1 for p in self.points)
        return positive or self.box is not None or self.mask_prompt is not None

    def points_array(self) -> Optional[np.ndarray]:
        if not self.points:
            return None
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)

    def labels_array(self) -> Optional[np.ndarray]:
        if not self.points:
            return None
        return np.array([p.label for p in self.points], dtype=np.int32)


@dataclass
class PropagationResult:
    """Holds all mask results after a full propagation run."""
    # {frame_index: {object_id: binary_mask (H, W, uint8)}}
    masks: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    # {frame_index: {object_id: confidence (0-1)}}
    confidence: dict[int, dict[int, float]] = field(default_factory=dict)
    total_frames: int = 0
    cancelled: bool = False


# ── Base class ─────────────────────────────────────────────────────────────

class SAM3RunnerBase:
    """Abstract interface for SAM3 inference."""

    def load_model(self) -> None:
        raise NotImplementedError

    def unload_model(self) -> None:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError

    def set_prompts(self, object_id: int, prompts: ObjectPrompts) -> None:
        raise NotImplementedError

    def get_prompts(self, object_id: int) -> ObjectPrompts:
        raise NotImplementedError

    def clear_object_prompts(self, object_id: int) -> None:
        raise NotImplementedError

    def clear_all_prompts(self) -> None:
        raise NotImplementedError

    def run_single_frame(
        self, frame_idx: int, frame: np.ndarray
    ) -> dict[int, np.ndarray]:
        """
        Run inference on a single frame with the current prompt state.

        Args:
            frame_idx: Frame index (used as conditioning frame in video state).
            frame:     BGR numpy array from OpenCV.

        Returns:
            Dict mapping object_id → binary mask (H, W, uint8, values 0/255).
        """
        raise NotImplementedError

    def propagate(
        self,
        frames: Iterator[tuple[int, np.ndarray]],
        start_frame_idx: int,
        total_frames: int,
        progress_callback: Optional[ProgressCallback] = None,
        mask_callback: Optional[MaskCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> PropagationResult:
        """
        Propagate prompts across all frames.

        Args:
            frames:            Iterator of (frame_idx, BGR array) tuples.
            start_frame_idx:   First frame index (for progress reporting).
            total_frames:      Total frames expected (for ETA calculation).
            progress_callback: Called with (current, total, fps, eta_s).
            mask_callback:     Called with (frame_idx, {obj_id: mask}) per frame.
            stop_event:        Set to abort mid-propagation.

        Returns:
            PropagationResult with all masks.
        """
        raise NotImplementedError

    @property
    def object_ids(self) -> list[int]:
        raise NotImplementedError


# ── Real SAM3 runner ───────────────────────────────────────────────────────

class SAM3Runner(SAM3RunnerBase):
    """
    SAM3 runner backed by the real sam2 library.

    Handles model loading, multi-object prompt management, live inference,
    propagation, and OOM recovery.
    """

    def __init__(
        self,
        gpu_info: GPUInfo,
        model_name: str = "sam3_large",
        models_dir: Path = MODELS_DIR,
    ) -> None:
        """
        Args:
            gpu_info:    Detected GPU backend and VRAM.
            model_name:  'sam3_large' or 'sam3_base'.
            models_dir:  Directory containing downloaded checkpoints.
        """
        self._gpu = gpu_info
        self._model_name = model_name
        self._models_dir = models_dir
        self._device_str = torch_device_string(gpu_info.backend)
        self._dtype_str = recommended_dtype(gpu_info.backend)

        self._predictor = None
        self._video_predictor = None
        self._inference_state = None
        self._lock = threading.Lock()

        # {object_id: ObjectPrompts}
        self._prompts: dict[int, ObjectPrompts] = {}

    # ── Model lifecycle ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the SAM3 checkpoint into device memory.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            RuntimeError:      On model load failure.
        """
        checkpoint = self._resolve_checkpoint()
        logger.info(
            "Loading %s on %s (%s)", checkpoint.name, self._device_str, self._dtype_str
        )

        try:
            import torch  # type: ignore[import-untyped]
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"SAM3 / sam2 not installed: {exc}. Run 'python install.py' to repair."
            ) from exc

        _CONFIG_MAP = {
            "sam3_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam3_base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
        }
        config_file = _CONFIG_MAP.get(self._model_name, _CONFIG_MAP["sam3_large"])

        dtype = getattr(torch, self._dtype_str)
        device = torch.device(self._device_str)

        self._video_predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=device,
        )

        # Image predictor shares the same loaded weights — no double load
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-untyped]
        self._predictor = SAM2ImagePredictor(self._video_predictor)

        logger.info("Model loaded successfully")

    def unload_model(self) -> None:
        """Release model from VRAM (call on window minimize)."""
        with self._lock:
            self._inference_state = None
            self._video_predictor = None
            self._predictor = None

        try:
            import torch  # type: ignore[import-untyped]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Model unloaded from VRAM")

    def is_loaded(self) -> bool:
        return self._video_predictor is not None

    def _resolve_checkpoint(self) -> Path:
        """
        Return the checkpoint path, auto-downgrading to Base if VRAM is low.

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If neither checkpoint exists.
        """
        name_map = {
            "sam3_large": SAM3_LARGE_FILENAME,
            "sam3_base": SAM3_BASE_FILENAME,
        }
        filename = name_map.get(self._model_name, SAM3_LARGE_FILENAME)

        # Auto-downgrade
        if (
            self._model_name == "sam3_large"
            and self._gpu.backend == Backend.CUDA
            and self._gpu.vram_gb < VRAM_THRESHOLD_LARGE_GB
        ):
            logger.warning(
                "VRAM %.1f GB < %.0f GB threshold; switching to SAM3-Base",
                self._gpu.vram_gb, VRAM_THRESHOLD_LARGE_GB,
            )
            filename = SAM3_BASE_FILENAME
            self._model_name = "sam3_base"

        path = self._models_dir / filename
        if not path.exists():
            # Try the other model as fallback
            fallback_name = SAM3_LARGE_FILENAME if filename == SAM3_BASE_FILENAME else SAM3_BASE_FILENAME
            fallback_path = self._models_dir / fallback_name
            if fallback_path.exists():
                logger.warning(
                    "Checkpoint %s not found; falling back to %s",
                    filename, fallback_name,
                )
                self._model_name = "sam3_large" if fallback_name == SAM3_LARGE_FILENAME else "sam3_base"
                return fallback_path
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. Run 'python install.py'."
            )
        return path

    # ── Prompt management ──────────────────────────────────────────────────

    def set_prompts(self, object_id: int, prompts: ObjectPrompts) -> None:
        with self._lock:
            self._prompts[object_id] = prompts

    def get_prompts(self, object_id: int) -> ObjectPrompts:
        return self._prompts.get(object_id, ObjectPrompts(object_id=object_id))

    def clear_object_prompts(self, object_id: int) -> None:
        with self._lock:
            self._prompts.pop(object_id, None)

    def clear_all_prompts(self) -> None:
        with self._lock:
            self._prompts.clear()
            self._inference_state = None

    @property
    def object_ids(self) -> list[int]:
        return list(self._prompts.keys())

    # ── Single-frame inference ─────────────────────────────────────────────

    def run_single_frame(
        self, frame_idx: int, frame: np.ndarray
    ) -> dict[int, np.ndarray]:
        """
        Run SAM3 on a single frame with the current prompt state.

        Uses SAM2ImagePredictor for fast single-frame inference without
        writing frames to disk.

        Returns:
            {object_id: binary mask uint8 (H, W)} for all prompted objects.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import cv2  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        # SAM2 expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results: dict[int, np.ndarray] = {}
        device_type = torch.device(self._device_str).type

        with self._lock:
            with torch.inference_mode():
                with torch.autocast(device_type=device_type, dtype=getattr(torch, self._dtype_str)):
                    self._predictor.set_image(rgb)

                    for obj_id, prompts in self._prompts.items():
                        if not prompts.has_prompts():
                            continue

                        pts = prompts.points_array()
                        lbs = prompts.labels_array()
                        box = prompts.box.as_numpy() if prompts.box else None

                        masks, _scores, _logits = self._predictor.predict(
                            point_coords=pts,
                            point_labels=lbs,
                            box=box,
                            multimask_output=False,
                        )
                        # masks: (1, H, W) bool → uint8 0/255
                        results[obj_id] = (masks[0].astype(np.uint8)) * 255

        return results

    # ── Propagation ────────────────────────────────────────────────────────

    def propagate(
        self,
        frames: Iterator[tuple[int, np.ndarray]],
        start_frame_idx: int,
        total_frames: int,
        progress_callback: Optional[ProgressCallback] = None,
        mask_callback: Optional[MaskCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> PropagationResult:
        """Full-video propagation with OOM recovery."""
        # Collect all frames upfront — the iterator is consumed once here, and
        # the buffer is reused across OOM retries without re-reading from disk.
        frame_buffer: list[np.ndarray] = []
        frame_indices: list[int] = []
        for idx, frame in frames:
            if stop_event and stop_event.is_set():
                result = PropagationResult(total_frames=total_frames)
                result.cancelled = True
                return result
            frame_buffer.append(frame)
            frame_indices.append(idx)

        if not frame_buffer:
            return PropagationResult(total_frames=total_frames)

        try:
            return self._run_inference(
                frame_buffer, frame_indices, start_frame_idx, total_frames,
                progress_callback, mask_callback, stop_event
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                return self._handle_oom(
                    frame_buffer, frame_indices, start_frame_idx, total_frames,
                    progress_callback, mask_callback, stop_event, exc
                )
            raise

    def _run_inference(
        self,
        frame_buffer: list[np.ndarray],
        frame_indices: list[int],
        start_frame_idx: int,
        total_frames: int,
        progress_callback: Optional[ProgressCallback],
        mask_callback: Optional[MaskCallback],
        stop_event: Optional[threading.Event],
    ) -> PropagationResult:
        import contextlib
        import os
        import tempfile

        import cv2  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        result = PropagationResult(total_frames=total_frames)
        t_start = time.monotonic()
        processed = 0

        device_type = torch.device(self._device_str).type

        # Compute scale factor from the first frame so prompt coordinates and
        # output masks stay aligned with the original resolution.
        _MAX_SIDE = 720
        orig_h, orig_w = frame_buffer[0].shape[:2]
        if orig_w > _MAX_SIDE or orig_h > _MAX_SIDE:
            scale = _MAX_SIDE / max(orig_w, orig_h)
        else:
            scale = 1.0
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)

        def _downscale(f: np.ndarray) -> np.ndarray:
            if scale == 1.0:
                return f
            return cv2.resize(f, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # torch.autocast on CPU only supports bfloat16/float16, not float32
        if device_type == "cpu":
            autocast_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        else:
            autocast_ctx = torch.autocast(
                device_type=device_type, dtype=getattr(torch, self._dtype_str)
            )

        with tempfile.TemporaryDirectory(prefix="sam3_prop_") as tmpdir:
            for i, frame in enumerate(frame_buffer):
                cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), _downscale(frame))

            with self._lock:
                with torch.inference_mode():
                    with autocast_ctx:
                        state = self._video_predictor.init_state(tmpdir)

                        # Map real frame index → temp sequence index so prompts
                        # land on the correct frame regardless of clip in-point.
                        idx_map = {real: ti for ti, real in enumerate(frame_indices)}

                        for obj_id, prompts in self._prompts.items():
                            if not prompts.has_prompts():
                                continue

                            pts = prompts.points_array()
                            if pts is not None and scale != 1.0:
                                pts = pts * scale

                            box = prompts.box.as_numpy() if prompts.box else None
                            if box is not None and scale != 1.0:
                                box = box * scale

                            prompt_temp_idx = idx_map.get(prompts.frame_idx, 0)

                            self._video_predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=prompt_temp_idx,
                                obj_id=obj_id,
                                points=pts,
                                labels=prompts.labels_array(),
                                box=box,
                            )

                        for out_frame_idx, out_obj_ids, out_logits in (
                            self._video_predictor.propagate_in_video(state)
                        ):
                            if stop_event and stop_event.is_set():
                                result.cancelled = True
                                break

                            real_idx = frame_indices[out_frame_idx]
                            frame_masks: dict[int, np.ndarray] = {}
                            frame_conf: dict[int, float] = {}

                            for i, obj_id in enumerate(out_obj_ids):
                                logit = out_logits[i, 0]
                                binary = (
                                    (logit > SAM3_LOGIT_THRESHOLD)
                                    .cpu().numpy()
                                    .astype(np.uint8) * 255
                                )
                                # Upscale mask back to original frame resolution
                                if scale != 1.0:
                                    binary = cv2.resize(
                                        binary, (orig_w, orig_h),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                frame_masks[int(obj_id)] = binary
                                conf = float(torch.sigmoid(logit).mean().cpu())
                                frame_conf[int(obj_id)] = conf

                            result.masks[real_idx] = frame_masks
                            result.confidence[real_idx] = frame_conf

                            if mask_callback:
                                mask_callback(real_idx, frame_masks)

                            processed += 1
                            elapsed = max(time.monotonic() - t_start, 0.001)
                            fps = processed / elapsed
                            eta = (total_frames - processed) / fps if fps > 0 else 0.0
                            if progress_callback:
                                progress_callback(processed, total_frames, fps, eta)

                        self._inference_state = state

        return result

    def _handle_oom(
        self,
        frame_buffer: list[np.ndarray],
        frame_indices: list[int],
        start_frame_idx: int,
        total_frames: int,
        progress_callback: Optional[ProgressCallback],
        mask_callback: Optional[MaskCallback],
        stop_event: Optional[threading.Event],
        original_exc: Exception,
    ) -> PropagationResult:
        """OOM recovery: downgrade model then retry, then CPU fallback."""
        try:
            import torch  # type: ignore[import-untyped]
            torch.cuda.empty_cache()
        except ImportError:
            pass

        if self._model_name != "sam3_base":
            base_path = self._models_dir / SAM3_BASE_FILENAME
            if base_path.exists():
                logger.warning("OOM: switching to SAM3-Base and retrying \u2026")
                self._model_name = "sam3_base"
                self.unload_model()
                try:
                    self.load_model()
                    return self._run_inference(
                        frame_buffer, frame_indices, start_frame_idx, total_frames,
                        progress_callback, mask_callback, stop_event
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
            else:
                logger.warning(
                    "OOM with Large model and SAM3-Base not downloaded. "
                    "Run 'python install.py' to download the Base checkpoint."
                )
                raise RuntimeError(
                    "VRAM insuficiente con el modelo Large. "
                    "Descarga SAM3-Base ejecutando 'python install.py'."
                ) from original_exc

        logger.warning("OOM even with Base model; falling back to CPU (slow) \u2026")
        from sam3_resolve.core.gpu_utils import GPUInfo
        cpu_gpu = GPUInfo(
            backend=Backend.CPU,
            device_name="CPU",
            vram_gb=0.0,
            driver_version="",
            cuda_version="",
        )
        self._gpu = cpu_gpu
        self._device_str = torch_device_string(Backend.CPU)
        self._dtype_str = recommended_dtype(Backend.CPU)
        self.unload_model()
        self.load_model()
        return self._run_inference(
            frame_buffer, frame_indices, start_frame_idx, total_frames,
            progress_callback, mask_callback, stop_event
        )


# ── Mock runner ────────────────────────────────────────────────────────────

class MockSAM3Runner(SAM3RunnerBase):
    """
    Drop-in SAM3 replacement that generates synthetic masks without GPU/model.

    Mask shape: filled ellipse centred on the first positive point (or box
    centre), sized proportionally to the frame.  During propagation the
    ellipse drifts slightly each frame to simulate realistic tracking.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._prompts: dict[int, ObjectPrompts] = {}
        self._load_delay_ms: float = 50.0    # simulated inference latency
        # Record all propagation calls for test assertions
        self.propagation_calls: list[dict] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load_model(self) -> None:
        time.sleep(self._load_delay_ms / 1000)
        self._loaded = True
        logger.info("[MOCK] SAM3Runner loaded")

    def unload_model(self) -> None:
        self._loaded = False
        logger.info("[MOCK] SAM3Runner unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Prompts ────────────────────────────────────────────────────────────

    def set_prompts(self, object_id: int, prompts: ObjectPrompts) -> None:
        self._prompts[object_id] = prompts

    def get_prompts(self, object_id: int) -> ObjectPrompts:
        return self._prompts.get(object_id, ObjectPrompts(object_id=object_id))

    def clear_object_prompts(self, object_id: int) -> None:
        self._prompts.pop(object_id, None)

    def clear_all_prompts(self) -> None:
        self._prompts.clear()

    @property
    def object_ids(self) -> list[int]:
        return list(self._prompts.keys())

    # ── Inference ──────────────────────────────────────────────────────────

    def run_single_frame(
        self, frame_idx: int, frame: np.ndarray
    ) -> dict[int, np.ndarray]:
        import time as _time
        _time.sleep(self._load_delay_ms / 1000)

        h, w = frame.shape[:2]
        results: dict[int, np.ndarray] = {}

        for obj_id, prompts in self._prompts.items():
            if not prompts.has_prompts():
                continue
            cx, cy = self._prompt_centre(prompts, w, h)
            results[obj_id] = self._make_ellipse_mask(h, w, cx, cy, drift=0)

        return results

    def propagate(
        self,
        frames: Iterator[tuple[int, np.ndarray]],
        start_frame_idx: int,
        total_frames: int,
        progress_callback: Optional[ProgressCallback] = None,
        mask_callback: Optional[MaskCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> PropagationResult:
        result = PropagationResult(total_frames=total_frames)
        t_start = time.monotonic()
        processed = 0

        call_record: dict = {
            "start_frame_idx": start_frame_idx,
            "total_frames": total_frames,
            "frames_processed": [],
        }
        self.propagation_calls.append(call_record)

        if stop_event and stop_event.is_set():
            result.cancelled = True
            call_record["cancelled"] = True
            return result

        for frame_idx, frame in frames:
            if stop_event and stop_event.is_set():
                result.cancelled = True
                call_record["cancelled"] = True
                break

            time.sleep(self._load_delay_ms / 1000)  # simulate per-frame time
            h, w = frame.shape[:2]
            frame_masks: dict[int, np.ndarray] = {}
            frame_conf: dict[int, float] = {}

            drift = (frame_idx - start_frame_idx) * 2   # 2px rightward per frame

            for obj_id, prompts in self._prompts.items():
                if not prompts.has_prompts():
                    continue
                cx, cy = self._prompt_centre(prompts, w, h)
                mask = self._make_ellipse_mask(h, w, cx, cy, drift=drift)
                frame_masks[obj_id] = mask
                frame_conf[obj_id] = 0.92  # mock confidence

            result.masks[frame_idx] = frame_masks
            result.confidence[frame_idx] = frame_conf
            call_record["frames_processed"].append(frame_idx)

            if mask_callback:
                mask_callback(frame_idx, frame_masks)

            processed += 1
            elapsed = max(time.monotonic() - t_start, 0.001)
            fps = processed / elapsed
            eta = (total_frames - processed) / fps if fps > 0 else 0.0
            if progress_callback:
                progress_callback(processed, total_frames, fps, eta)

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _prompt_centre(
        prompts: ObjectPrompts, frame_w: int, frame_h: int
    ) -> tuple[float, float]:
        """Return (cx, cy) derived from prompts."""
        if prompts.box:
            b = prompts.box
            return (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2

        positive = [p for p in prompts.points if p.label == 1]
        if positive:
            xs = [p.x for p in positive]
            ys = [p.y for p in positive]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        return frame_w / 2, frame_h / 2

    @staticmethod
    def _make_ellipse_mask(
        h: int, w: int, cx: float, cy: float, drift: int = 0
    ) -> np.ndarray:
        """
        Draw a filled white ellipse on a black background.

        The ellipse covers roughly 20% of the frame area, mimicking a
        realistic object segmentation.
        """
        import cv2  # type: ignore[import-untyped]

        mask = np.zeros((h, w), dtype=np.uint8)
        rx = max(10, int(w * 0.12))
        ry = max(10, int(h * 0.18))
        centre = (int(cx) + drift, int(cy))
        cv2.ellipse(mask, centre, (rx, ry), 0, 0, 360, 255, -1)
        return mask


# ── Confidence helpers ─────────────────────────────────────────────────────

def confidence_level(confidence: float) -> str:
    """
    Categorise a float confidence score into a display tier.

    Args:
        confidence: Float in [0, 1].

    Returns:
        One of 'high', 'medium', 'low', 'none'.
    """
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.40:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "none"


def mask_is_empty(mask: np.ndarray) -> bool:
    """Return True if the mask contains no foreground pixels."""
    return bool(mask.max() == 0)


# ── Factory ────────────────────────────────────────────────────────────────

def create_runner(
    gpu_info: Optional[GPUInfo] = None,
    model_name: Optional[str] = None,
    force_mock: bool = False,
) -> SAM3RunnerBase:
    """
    Return a SAM3Runner for the current environment.

    Falls back to MockSAM3Runner if:
      - force_mock is True
      - torch / sam2 cannot be imported
      - GPU is insufficient

    Args:
        gpu_info:   Detected GPU. Auto-detected if None.
        model_name: 'sam3_large' or 'sam3_base'.
        force_mock: Skip real runner entirely.

    Returns:
        A SAM3RunnerBase instance (real or mock).
    """
    if force_mock:
        return MockSAM3Runner()

    # Read model preference from config if not explicitly passed
    if model_name is None:
        try:
            from sam3_resolve.config import Config
            cfg_model = Config.instance().get("model_name", "large")
            model_name = "sam3_large" if cfg_model == "large" else "sam3_base"
        except Exception:  # noqa: BLE001
            model_name = "sam3_large"

    try:
        if gpu_info is None:
            from sam3_resolve.core.gpu_utils import detect_gpu
            gpu_info = detect_gpu()
        import torch  # type: ignore[import-untyped]  # noqa: F401
        import sam2  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, RuntimeError):
        logger.warning("torch/sam2 not available; using MockSAM3Runner")
        return MockSAM3Runner()

    return SAM3Runner(gpu_info=gpu_info, model_name=model_name)
