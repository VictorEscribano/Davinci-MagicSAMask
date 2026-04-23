"""
SAM3 Resolve Plugin — entry point.

Normal mode (launched by DaVinci Resolve as a Comp Script):
    python plugin_main.py
    — Connects to running Resolve, reads the selected clip, starts inference.

Debug mode (standalone, no Resolve required):
    python -m sam3_resolve.plugin_main --debug [--file /path/to/video.mp4]
    — Opens a file-picker dialog (or uses --file directly), loads the video
      with OpenCV, and runs the full SAM3 UI with MockSAM3Runner.
    — No Resolve installation needed.

Steps (normal):
  1. Detect OS and add Resolve API to sys.path.
  2. Activate venv site-packages (so torch / sam2 are importable).
  3. Check deps; launch setup_wizard if missing.
  4. Start QApplication + MainWindow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Venv bootstrap ─────────────────────────────────────────────────────────

def _activate_venv() -> None:
    """Prepend the plugin venv's site-packages to sys.path."""
    try:
        from sam3_resolve.config import Config
        cfg = Config.instance()
        venv_path = cfg.get("venv_path", "")
        if not venv_path:
            return
        import platform
        if platform.system() == "Windows":
            sp = Path(venv_path) / "Lib" / "site-packages"
        else:
            sp = next(
                (Path(venv_path) / "lib" / d.name / "site-packages"
                 for d in (Path(venv_path) / "lib").iterdir()
                 if d.name.startswith("python")),
                None,
            )
        if sp and sp.exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
            logger.debug("Activated venv site-packages: %s", sp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not activate venv: %s", exc)


# ── cv2 frame reader ───────────────────────────────────────────────────────

class _Cv2FrameReader:
    """
    Thin wrapper around cv2.VideoCapture for debug mode.

    Provides a callable  reader(frame_idx) -> BGR ndarray | None
    that the MainWindow uses via attach_frame_reader().
    """

    def __init__(self, path: str) -> None:
        import cv2
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width        = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_idx    = -1

    def __call__(self, frame_idx: int) -> Optional[np.ndarray]:
        import cv2
        # Only seek if the requested frame isn't the next sequential one
        if frame_idx != self._last_idx + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if ok:
            self._last_idx = frame_idx
            return frame
        return None

    def release(self) -> None:
        self._cap.release()


# ── Debug mode entry point ─────────────────────────────────────────────────

def debug_main(file_path: Optional[str] = None, use_mock: bool = False) -> None:
    """
    Launch the full SAM3 UI without DaVinci Resolve.

    If file_path is None, opens a QFileDialog to pick a video.
    By default uses the real SAM runner (requires model downloaded).
    Pass use_mock=True (or --mock flag) to skip GPU/model entirely.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog
    except ImportError:
        print("PyQt6 not found. Run: pip install PyQt6", file=sys.stderr)
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)

    # ── Pick video file ──────────────────────────────────────────────────
    if not file_path:
        chosen, _ = QFileDialog.getOpenFileName(
            None,
            "Open video file — SAM3 debug mode",
            str(Path.home()),
            "Video files (*.mp4 *.mov *.mkv *.avi *.m4v *.mxf);;All files (*)",
        )
        if not chosen:
            logger.info("No file selected — exiting debug mode.")
            sys.exit(0)
        file_path = chosen

    # ── Open with OpenCV ──────────────────────────────────────────────────
    try:
        reader = _Cv2FrameReader(file_path)
    except IOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logger.info(
        "[DEBUG] Opened %s — %dx%d, %.3f fps, %d frames",
        Path(file_path).name, reader.width, reader.height,
        reader.fps, reader.total_frames,
    )

    # ── Build a synthetic ClipInfo from the video metadata ────────────────
    from sam3_resolve.core.resolve_bridge import ClipInfo, ClipFormat
    clip = ClipInfo(
        name=Path(file_path).name,
        file_path=file_path,
        proxy_path="",
        media_pool_uuid="debug-0001",
        color_label="",
        width=reader.width,
        height=reader.height,
        fps=reader.fps,
        duration_frames=reader.total_frames,
        start_frame=0,
        end_frame=reader.total_frames,
        in_point_frame=0,
        out_point_frame=reader.total_frames,
        start_timecode="00:00:00:00",
        clip_format=ClipFormat.DIRECT,
        track_index=0,
    )

    # ── Build runner ──────────────────────────────────────────────────────
    from sam3_resolve.core.gpu_utils import detect_gpu
    from sam3_resolve.core.sam3_runner import MockSAM3Runner, create_runner
    if use_mock:
        runner = MockSAM3Runner()
        gpu_label = "[DEBUG] MockSAM3Runner — no GPU needed"
        gpu_ready = True
    else:
        gpu = detect_gpu()
        runner = create_runner(gpu_info=gpu)
        if gpu.backend.value == "cuda":
            gpu_label = f"[DEBUG] {gpu.device_name} · CUDA · {gpu.vram_gb:.0f}GB"
        elif gpu.backend.value == "mps":
            gpu_label = "[DEBUG] Apple Silicon · MPS"
        else:
            gpu_label = "[DEBUG] CPU mode"
        gpu_ready = gpu.backend.value != "cpu"

    # ── Launch window ─────────────────────────────────────────────────────
    from sam3_resolve.ui.main_window import MainWindow
    window = MainWindow(clip=clip, runner=runner)
    window.set_gpu_info_label(gpu_label, ready=gpu_ready)
    window.attach_frame_reader(reader)
    window.show()

    ret = app.exec()
    reader.release()
    sys.exit(ret)


# ── Normal (Resolve) entry point ───────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _activate_venv()

    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print(
            "PyQt6 not found. Run 'python install.py' to install dependencies.",
            file=sys.stderr,
        )
        sys.exit(1)

    from sam3_resolve.core.resolve_bridge import create_bridge
    from sam3_resolve.core.gpu_utils import detect_gpu
    from sam3_resolve.core.sam3_runner import create_runner
    from sam3_resolve.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)

    bridge = create_bridge()
    gpu = detect_gpu()

    def _load_clip_from_resolve():
        """Fetch the current Resolve clip and return (ClipInfo, frame_reader)."""
        clip = bridge.get_current_clip()  # raises if nothing selected
        if not Path(clip.file_path).exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {clip.file_path}\n"
                "Verifica que el clip no esté offline en Resolve."
            )
        reader = _Cv2FrameReader(clip.file_path)
        return clip, reader

    clip = None
    initial_reader = None
    try:
        clip, initial_reader = _load_clip_from_resolve()
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo cargar el clip al iniciar: %s", exc)

    runner = create_runner(gpu_info=gpu)
    window = MainWindow(clip=clip, runner=runner)

    if gpu.backend.value == "cuda":
        label = f"{gpu.device_name} · CUDA · {gpu.vram_gb:.0f}GB   GPU Ready"
        window.set_gpu_info_label(label, ready=True)
    elif gpu.backend.value == "mps":
        window.set_gpu_info_label("Apple Silicon · MPS   GPU Ready", ready=True)
    else:
        window.set_gpu_info_label("CPU mode (no GPU detected)", ready=False)

    # Connect the "Cargar Clip" button so the user can retry at any time
    window.set_clip_loader(_load_clip_from_resolve)

    if initial_reader is not None:
        window.attach_frame_reader(initial_reader)
    else:
        window.bottom_panel.append_log(
            "WARN",
            "No hay clip seleccionado en Resolve. "
            "Selecciona un clip en el timeline y pulsa '⟳ Cargar Clip'.",
        )

    window.show()
    sys.exit(app.exec())


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Mask Tracker")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run without DaVinci Resolve (opens a file picker)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Video file to open directly in debug mode (skips file picker)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockSAM3Runner in debug mode (no GPU or model needed)",
    )
    args = parser.parse_args()

    if args.debug or args.file:
        debug_main(file_path=args.file, use_mock=args.mock)
    else:
        main()
