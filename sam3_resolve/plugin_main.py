"""
SAM3 Resolve Plugin — entry point.

Registered as a Resolve Comp Script. Steps:
  1. Detect OS and add Resolve API to sys.path.
  2. Activate venv site-packages (so torch / sam2 are importable).
  3. Check deps; launch setup_wizard if missing.
  4. Start QApplication + MainWindow.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
                (Path(venv_path) / "lib" / d / "site-packages"
                 for d in Path(venv_path).joinpath("lib").iterdir()
                 if d.name.startswith("python")),
                None,
            )
        if sp and sp.exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
            logger.debug("Activated venv site-packages: %s", sp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not activate venv: %s", exc)


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
    from sam3_resolve.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)

    bridge = create_bridge()
    gpu = detect_gpu()

    # Try to get current clip; fall back to no-clip mode
    clip = None
    try:
        clip = bridge.get_current_clip()
    except Exception as exc:  # noqa: BLE001
        logger.info("No clip available at startup: %s", exc)

    window = MainWindow(clip=clip)

    # Show GPU status
    if gpu.backend.value == "cuda":
        label = f"{gpu.device_name} · CUDA · {gpu.vram_gb:.0f}GB   GPU Ready"
        window.set_gpu_info_label(label, ready=True)
    elif gpu.backend.value == "mps":
        window.set_gpu_info_label("Apple Silicon · MPS   GPU Ready", ready=True)
    else:
        window.set_gpu_info_label("CPU mode (no GPU detected)", ready=False)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
