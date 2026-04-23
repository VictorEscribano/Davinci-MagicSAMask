"""
QThread workers for SAM3 inference.

LiveInferenceWorker  — debounced single-frame inference (live preview)
PropagationWorker    — full-video propagation with per-frame progress

Both workers are pure QThread subclasses; they call SAM3RunnerBase methods
which are pure Python and independently testable without PyQt6.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from sam3_resolve.core.sam3_runner import SAM3RunnerBase
from sam3_resolve.core.media_handler import MediaHandler

logger = logging.getLogger(__name__)


class LiveInferenceWorker(QThread):
    """
    Runs single-frame SAM3 inference in a background thread.

    Debouncing is handled externally via QTimer.singleShot in the canvas;
    this worker just runs and reports back.

    Signals:
        mask_ready(dict):   {object_id: np.ndarray} mask per object
        error_occurred(str): error message
    """

    mask_ready = pyqtSignal(object)     # dict[int, np.ndarray]
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        runner: SAM3RunnerBase,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._frame_idx: int = 0
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def set_task(self, frame_idx: int, frame: np.ndarray) -> None:
        """Update the frame to run inference on (call before start())."""
        with self._lock:
            self._frame_idx = frame_idx
            self._frame = frame.copy()

    def run(self) -> None:
        with self._lock:
            frame_idx = self._frame_idx
            frame = self._frame

        if frame is None:
            return

        try:
            masks = self._runner.run_single_frame(frame_idx, frame)
            self.mask_ready.emit(masks)
        except Exception as exc:  # noqa: BLE001
            logger.error("LiveInferenceWorker error: %s", exc)
            self.error_occurred.emit(str(exc))


class PropagationWorker(QThread):
    """
    Runs full-video SAM3 propagation in a background thread.

    Signals:
        progress_updated(int, int, float, float): current, total, fps, eta_s
        frame_masks_ready(int, object):           frame_idx, dict[int, np.ndarray]
        finished(object):                         PropagationResult
        error_occurred(str):                      error message
        cancelled():                              user cancelled
    """

    progress_updated = pyqtSignal(int, int, float, float)
    frame_masks_ready = pyqtSignal(int, object)   # frame_idx, dict
    finished = pyqtSignal(object)                 # PropagationResult
    error_occurred = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        runner: SAM3RunnerBase,
        media_handler: MediaHandler,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._media = media_handler
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._stop_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Propagation stops at the next frame boundary."""
        self._stop_event.set()

    def run(self) -> None:
        clip = self._media.clip
        start = self._start_frame if self._start_frame is not None else clip.in_point_frame
        end = self._end_frame if self._end_frame is not None else clip.out_point_frame
        total = end - start

        try:
            frames_iter = self._media.iter_frames(
                start=start, end=end, stop_event=self._stop_event
            )

            result = self._runner.propagate(
                frames=frames_iter,
                start_frame_idx=start,
                total_frames=total,
                progress_callback=self._on_progress,
                mask_callback=self._on_mask,
                stop_event=self._stop_event,
            )

            if result.cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001
            logger.error("PropagationWorker error: %s", exc)
            self.error_occurred.emit(str(exc))

    def _on_progress(
        self, current: int, total: int, fps: float, eta_s: float
    ) -> None:
        self.progress_updated.emit(current, total, fps, eta_s)

    def _on_mask(self, frame_idx: int, masks: dict) -> None:
        self.frame_masks_ready.emit(frame_idx, masks)
