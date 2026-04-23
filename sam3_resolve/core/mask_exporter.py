"""
Mask exporter — writes 16-bit alpha PNG sequences for Resolve import.

Directory layout:
    <output_dir>/
        object_001/
            frame_000000.png
            frame_000001.png
            …
        object_002/
            …
        manifest.json   ← sidecar read by the Resolve bridge

Each PNG is a single-channel uint16 image (0 = transparent, 65535 = opaque).
Feathering and upscaling are applied here (not in the propagation step) so the
original binary masks are kept unmodified for re-export with different settings.

The work is split across a multiprocessing.Pool so that PNG encoding (which is
CPU-bound and releases the GIL via zlib) runs in parallel.  Pool is only used
when more than EXPORT_SERIAL_THRESHOLD frames are present; below that threshold
the overhead isn't worth it.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from sam3_resolve.constants import EXPORT_PNG_BIT_DEPTH, EXPORT_WORKERS
from sam3_resolve.core.sam3_runner import PropagationResult

logger = logging.getLogger(__name__)

EXPORT_SERIAL_THRESHOLD = 30   # frames; below this use single-process


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class ExportSettings:
    """Controls how masks are written to disk."""
    output_dir: Path
    original_width: int    = 1920
    original_height: int   = 1080
    feather_px: int        = 0        # additional gaussian blur radius (px, original res)
    object_names: dict[int, str] = field(default_factory=dict)   # display names for manifest


@dataclass
class ExportResult:
    frames_written: int = 0
    total_frames:   int = 0
    output_dir:     Path = Path(".")
    cancelled:      bool = False
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and not self.cancelled


# ── Per-frame encode (top-level so multiprocessing can pickle it) ──────────

def _encode_frame(args: tuple) -> tuple[int, int]:
    """
    Write a single mask PNG.  Called by both the serial path and Pool workers.

    Returns (object_id, frame_idx) on success; raises on failure.
    """
    obj_id, frame_idx, mask_8bit, out_path_str, orig_h, orig_w, feather_px = args
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Upscale if needed
    h, w = mask_8bit.shape
    if h != orig_h or w != orig_w:
        mask_8bit = cv2.resize(
            mask_8bit, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )

    # 2. Feather (Gaussian blur on 8-bit, then re-threshold edges)
    if feather_px > 0:
        sigma = feather_px * 0.5
        mask_8bit = cv2.GaussianBlur(mask_8bit, (0, 0), sigma)

    # 3. Convert to 16-bit
    mask_16 = (mask_8bit.astype(np.uint32) * 257).clip(0, 65535).astype(np.uint16)

    # 4. Write PNG
    cv2.imwrite(str(out_path), mask_16)
    return obj_id, frame_idx


# ── Main exporter class ─────────────────────────────────────────────────────

class MaskExporter:
    """
    Synchronous (blocking) mask exporter.

    Designed to run inside an ExportWorker QThread; call export() from run().

    Usage::

        exporter = MaskExporter(result, settings, progress_callback)
        export_result = exporter.export()
    """

    def __init__(
        self,
        result: PropagationResult,
        settings: ExportSettings,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_event: Optional[object] = None,   # threading.Event
    ) -> None:
        self._result   = result
        self._settings = settings
        self._progress = progress_callback
        self._stop     = stop_event

    def export(self) -> ExportResult:
        s = self._settings
        r = self._result
        out_dir = Path(s.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect all (obj_id, frame_idx, mask) tuples
        tasks: list[tuple] = []
        for frame_idx, obj_masks in sorted(r.masks.items()):
            for obj_id, mask in obj_masks.items():
                if mask is None:
                    continue
                obj_dir = out_dir / f"object_{obj_id:03d}"
                png_path = obj_dir / f"frame_{frame_idx:06d}.png"
                tasks.append((
                    obj_id, frame_idx, mask,
                    str(png_path),
                    s.original_height, s.original_width,
                    s.feather_px,
                ))

        total = len(tasks)
        result = ExportResult(total_frames=total, output_dir=out_dir)

        if total == 0:
            self._write_manifest(out_dir, {}, 0)
            return result

        written = 0
        try:
            if total <= EXPORT_SERIAL_THRESHOLD:
                for task in tasks:
                    if self._cancelled():
                        result.cancelled = True
                        return result
                    _encode_frame(task)
                    written += 1
                    self._report(written, total)
            else:
                with multiprocessing.Pool(processes=EXPORT_WORKERS) as pool:
                    for _ in pool.imap_unordered(_encode_frame, tasks, chunksize=4):
                        if self._cancelled():
                            pool.terminate()
                            result.cancelled = True
                            return result
                        written += 1
                        self._report(written, total)

        except Exception as exc:
            logger.error("MaskExporter error: %s", exc)
            result.error = str(exc)
            return result

        result.frames_written = written
        self._write_manifest(out_dir, self._object_frame_counts(), total)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cancelled(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    def _report(self, done: int, total: int) -> None:
        if self._progress:
            self._progress(done, total)

    def _object_frame_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for obj_masks in self._result.masks.values():
            for obj_id in obj_masks:
                counts[obj_id] = counts.get(obj_id, 0) + 1
        return counts

    def _write_manifest(
        self,
        out_dir: Path,
        obj_counts: dict[int, int],
        total_tasks: int,
    ) -> None:
        s = self._settings
        objects = []
        for obj_id, count in sorted(obj_counts.items()):
            objects.append({
                "object_id":   obj_id,
                "name":        s.object_names.get(obj_id, f"Object {obj_id}"),
                "folder":      f"object_{obj_id:03d}",
                "frame_count": count,
                "bit_depth":   EXPORT_PNG_BIT_DEPTH,
            })

        manifest = {
            "version":          1,
            "exported_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_tasks":      total_tasks,
            "original_width":   s.original_width,
            "original_height":  s.original_height,
            "feather_px":       s.feather_px,
            "objects":          objects,
        }

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Manifest written to %s", manifest_path)
