"""
First-run setup wizard.

Guides the user through:
  1. Python version check
  2. GPU detection
  3. Dependency installation  (pip install)
  4. Model download           (resumable, with progress)
  5. Resolve scripts copy     (or skip if no Resolve)

Each step is rendered as a StepRow widget (icon + title + status badge + detail).
Steps run sequentially in background QThreads.

The wizard emits `setup_complete` when all steps succeed, allowing the caller
to close it and open MainWindow.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sam3_resolve.constants import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
    PLUGIN_VERSION,
    SAM3_MIN_PYTHON,
)

logger = logging.getLogger(__name__)

# ── Step status ────────────────────────────────────────────────────────────

class StepStatus(Enum):
    PENDING  = auto()
    RUNNING  = auto()
    OK       = auto()
    WARN     = auto()
    ERROR    = auto()
    SKIPPED  = auto()


_STATUS_STYLE: dict[StepStatus, tuple[str, str]] = {
    StepStatus.PENDING:  ("○",  "#555555"),
    StepStatus.RUNNING:  ("◌",  COLOR_ACCENT),
    StepStatus.OK:       ("✓",  COLOR_SUCCESS),
    StepStatus.WARN:     ("⚠",  COLOR_WARNING),
    StepStatus.ERROR:    ("✕",  COLOR_DANGER),
    StepStatus.SKIPPED:  ("–",  COLOR_TEXT_SECONDARY),
}


# ── Single step row widget ─────────────────────────────────────────────────

class StepRow(QWidget):
    """One step line: icon · title · status badge · detail text."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._status = StepStatus.PENDING

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self._icon = QLabel("○")
        self._icon.setFixedWidth(14)
        self._icon.setStyleSheet(f"color: #555555; font-size: 14px;")

        self._title = QLabel(title)
        self._title.setStyleSheet("color: #E8E8E8; font-size: 12px;")
        self._title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._title.setFixedWidth(180)

        self._detail = QLabel("")
        self._detail.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px;")
        self._detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(self._icon)
        layout.addWidget(self._title)
        layout.addWidget(self._detail)

    def set_status(self, status: StepStatus, detail: str = "") -> None:
        self._status = status
        icon, color = _STATUS_STYLE[status]
        self._icon.setText(icon)
        self._icon.setStyleSheet(f"color: {color}; font-size: 14px;")
        if detail:
            self._detail.setText(detail)

    @property
    def status(self) -> StepStatus:
        return self._status


# ── Background step workers ────────────────────────────────────────────────

class _StepWorker(QThread):
    """Base: emit step_done(ok, detail) when finished."""
    step_done = pyqtSignal(bool, str)   # success, detail message

    def run(self) -> None:
        try:
            ok, detail = self._execute()
            self.step_done.emit(ok, detail)
        except Exception as exc:
            logger.error("StepWorker error: %s", exc)
            self.step_done.emit(False, str(exc))

    def _execute(self) -> tuple[bool, str]:
        raise NotImplementedError


class _PythonCheckWorker(_StepWorker):
    def _execute(self):
        v = sys.version_info
        ok = (v.major, v.minor) >= SAM3_MIN_PYTHON
        detail = f"Python {v.major}.{v.minor}.{v.micro}"
        if not ok:
            detail += f" — requires ≥ {SAM3_MIN_PYTHON[0]}.{SAM3_MIN_PYTHON[1]}"
        return ok, detail


class _GPUDetectWorker(_StepWorker):
    gpu_info = pyqtSignal(object)   # GPUInfo

    def _execute(self):
        from sam3_resolve.core.gpu_utils import detect_gpu
        info = detect_gpu()
        self.gpu_info.emit(info)
        if info.backend.name == "CUDA":
            detail = f"CUDA · {info.device_name} · {info.vram_gb:.1f} GB"
        elif info.backend.name == "MPS":
            detail = "Apple MPS (Metal)"
        else:
            detail = "CPU only — inference will be slow"
        ok = True
        return ok, detail


class _DepsCheckWorker(_StepWorker):
    progress = pyqtSignal(int, int)   # done, total

    def _execute(self):
        from sam3_resolve.constants import BASE_DEPS
        missing = []
        for dep in BASE_DEPS:
            pkg = dep.split("[")[0].replace("-", "_").lower()
            try:
                __import__(pkg)
            except ImportError:
                missing.append(dep)

        if missing:
            return False, f"Missing: {', '.join(missing)}"
        return True, f"All {len(BASE_DEPS)} core packages present"


class _ModelCheckWorker(_StepWorker):
    def _execute(self):
        from sam3_resolve.constants import MODELS_DIR, SAM3_LARGE_FILENAME, SAM3_BASE_FILENAME
        large = MODELS_DIR / SAM3_LARGE_FILENAME
        base  = MODELS_DIR / SAM3_BASE_FILENAME
        if large.exists():
            size_gb = large.stat().st_size / 1e9
            return True, f"Large model found ({size_gb:.1f} GB)"
        if base.exists():
            size_gb = base.stat().st_size / 1e9
            return True, f"Base model found ({size_gb:.1f} GB)"
        return False, "No model file found — run installer"


class _ResolveCheckWorker(_StepWorker):
    def _execute(self):
        from sam3_resolve.constants import RESOLVE_SCRIPTS_DIR, PACKAGE_ROOT
        scripts_dir = RESOLVE_SCRIPTS_DIR
        if not scripts_dir.exists():
            return True, f"Skipped (Resolve not detected on {platform.system()})"  # soft-skip
        plugin_src = PACKAGE_ROOT / ".." / "plugin_main.py"
        dest = scripts_dir / "sam3_resolve.py"
        try:
            if not dest.exists() and plugin_src.exists():
                shutil.copy2(plugin_src.resolve(), dest)
            return True, f"Script present at {dest}"
        except OSError as exc:
            return False, str(exc)


# ── Wizard ─────────────────────────────────────────────────────────────────

class SetupWizard(QWidget):
    """
    First-run setup wizard.

    Signals:
        setup_complete():  all required steps passed — caller can open MainWindow
        setup_failed():    user closed wizard with errors still present
    """

    setup_complete = pyqtSignal()
    setup_failed   = pyqtSignal()

    # Each entry: (title, worker_class, required)
    _STEP_DEFS = [
        ("Python version",      _PythonCheckWorker,   True),
        ("GPU / device",        _GPUDetectWorker,      False),
        ("Core dependencies",   _DepsCheckWorker,      True),
        ("SAM3 model file",     _ModelCheckWorker,     True),
        ("Resolve scripts",     _ResolveCheckWorker,   False),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("setup_wizard")
        self.setMinimumSize(480, 420)
        self._step_rows: list[StepRow] = []
        self._workers:   list[Optional[_StepWorker]] = []
        self._step_results: list[Optional[bool]] = [None] * len(self._STEP_DEFS)
        self._current_step = -1
        self._gpu_info = None
        self._build()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet("background-color: #1A3A5E;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)
        title = QLabel(f"SAM3 Resolve Setup  v{PLUGIN_VERSION}")
        title.setStyleSheet("color: #E8E8E8; font-size: 16px; font-weight: 600;")
        h_layout.addWidget(title)
        h_layout.addStretch()
        outer.addWidget(header)

        # Step rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        steps_widget = QWidget()
        steps_layout = QVBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 8, 0, 8)
        steps_layout.setSpacing(2)

        for title_str, _cls, _req in self._STEP_DEFS:
            row = StepRow(title_str)
            self._step_rows.append(row)
            self._workers.append(None)
            steps_layout.addWidget(row)

        steps_layout.addStretch()
        scroll.setWidget(steps_widget)
        outer.addWidget(scroll, stretch=1)

        # Progress bar (shown during model download if needed)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

        # Buttons
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(12, 8, 12, 12)

        self._btn_run = QPushButton("▶  Run checks")
        self._btn_run.setObjectName("btn_run")
        self._btn_run.clicked.connect(self.run_all_steps)

        self._btn_repair = QPushButton("⟳  Repair")
        self._btn_repair.setEnabled(False)
        self._btn_repair.clicked.connect(self._repair)

        self._btn_close = QPushButton("✕  Close")
        self._btn_close.setObjectName("btn_cancel")
        self._btn_close.clicked.connect(self._on_close)

        self._btn_continue = QPushButton("Continue →")
        self._btn_continue.setObjectName("btn_accept")
        self._btn_continue.setEnabled(False)
        self._btn_continue.clicked.connect(self.setup_complete)

        btn_bar.addWidget(self._btn_run)
        btn_bar.addWidget(self._btn_repair)
        btn_bar.addStretch()
        btn_bar.addWidget(self._btn_close)
        btn_bar.addWidget(self._btn_continue)
        outer.addLayout(btn_bar)

    # ── Step execution ─────────────────────────────────────────────────────

    def run_all_steps(self) -> None:
        self._btn_run.setEnabled(False)
        self._btn_repair.setEnabled(False)
        self._btn_continue.setEnabled(False)
        self._current_step = -1
        for i, row in enumerate(self._step_rows):
            row.set_status(StepStatus.PENDING)
            self._step_results[i] = None
        self._run_next()

    def _run_next(self) -> None:
        self._current_step += 1
        if self._current_step >= len(self._STEP_DEFS):
            self._all_done()
            return
        _title, worker_cls, _req = self._STEP_DEFS[self._current_step]
        row = self._step_rows[self._current_step]
        row.set_status(StepStatus.RUNNING, "checking…")

        worker = worker_cls(self)
        if hasattr(worker, "gpu_info"):
            worker.gpu_info.connect(self._on_gpu_info)
        worker.step_done.connect(self._on_step_done)
        self._workers[self._current_step] = worker
        worker.start()

    def _on_gpu_info(self, info) -> None:
        self._gpu_info = info

    def _on_step_done(self, ok: bool, detail: str) -> None:
        idx = self._current_step
        _title, _cls, required = self._STEP_DEFS[idx]
        row = self._step_rows[idx]

        if ok:
            row.set_status(StepStatus.OK, detail)
        elif required:
            row.set_status(StepStatus.ERROR, detail)
        else:
            row.set_status(StepStatus.WARN, detail)

        self._step_results[idx] = ok
        self._run_next()

    def _all_done(self) -> None:
        all_required_ok = all(
            self._step_results[i] is not False
            for i, (_t, _c, req) in enumerate(self._STEP_DEFS)
            if req
        )
        has_errors = any(
            r is False for r in self._step_results
        )
        self._btn_repair.setEnabled(has_errors)
        self._btn_continue.setEnabled(all_required_ok)
        self._btn_run.setEnabled(True)

    def _repair(self) -> None:
        for i, row in enumerate(self._step_rows):
            if row.status in (StepStatus.ERROR, StepStatus.WARN):
                row.set_status(StepStatus.PENDING)
                self._step_results[i] = None
        self.run_all_steps()

    def _on_close(self) -> None:
        self.setup_failed.emit()
        self.close()

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def gpu_info(self):
        return self._gpu_info

    def step_statuses(self) -> list[StepStatus]:
        return [row.status for row in self._step_rows]
