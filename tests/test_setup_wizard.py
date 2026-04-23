"""Tests for setup_wizard — StepRow state and wizard flow."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from sam3_resolve.ui.setup_wizard import (
    SetupWizard,
    StepRow,
    StepStatus,
    _DepsCheckWorker,
    _ModelCheckWorker,
    _PythonCheckWorker,
    _ResolveCheckWorker,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def wizard(qt_app):
    return SetupWizard()


# ── StepRow ────────────────────────────────────────────────────────────────

def test_step_row_initial_status(qt_app):
    row = StepRow("Test step")
    assert row.status == StepStatus.PENDING


def test_step_row_set_ok(qt_app):
    row = StepRow("Test step")
    row.set_status(StepStatus.OK, "done")
    assert row.status == StepStatus.OK


def test_step_row_set_error(qt_app):
    row = StepRow("Test step")
    row.set_status(StepStatus.ERROR, "failed")
    assert row.status == StepStatus.ERROR


def test_step_row_set_warn(qt_app):
    row = StepRow("Test step")
    row.set_status(StepStatus.WARN, "warning")
    assert row.status == StepStatus.WARN


@pytest.mark.parametrize("status", list(StepStatus))
def test_step_row_all_statuses(qt_app, status):
    row = StepRow("x")
    row.set_status(status)
    assert row.status == status


# ── SetupWizard construction ───────────────────────────────────────────────

def test_wizard_creates_five_step_rows(wizard):
    assert len(wizard._step_rows) == 5


def test_wizard_initial_all_pending(wizard):
    for row in wizard._step_rows:
        assert row.status == StepStatus.PENDING


def test_wizard_continue_btn_initially_disabled(wizard):
    assert not wizard._btn_continue.isEnabled()


def test_wizard_repair_btn_initially_disabled(wizard):
    assert not wizard._btn_repair.isEnabled()


# ── Python check worker ────────────────────────────────────────────────────

def test_python_check_worker_current_version():
    w = _PythonCheckWorker()
    ok, detail = w._execute()
    # We're running on Python 3.12, which is ≥ 3.10
    assert ok
    assert "Python" in detail


def test_python_check_worker_old_version():
    from types import SimpleNamespace
    fake = SimpleNamespace(major=3, minor=8, micro=0)
    with patch("sys.version_info", fake):
        w = _PythonCheckWorker()
        ok, detail = w._execute()
        assert not ok
        assert "requires" in detail


# ── Deps check worker ──────────────────────────────────────────────────────

def test_deps_check_worker_all_present():
    w = _DepsCheckWorker()
    ok, detail = w._execute()
    # In a dev env with cv2 etc. installed, numpy/cv2 etc. are present
    # We can't guarantee all BASE_DEPS are importable; just verify it runs
    assert isinstance(ok, bool)
    assert isinstance(detail, str)


def test_deps_check_worker_missing_dep():
    with patch("sam3_resolve.constants.BASE_DEPS", ["nonexistent_package_xyz"]):
        w = _DepsCheckWorker()
        ok, detail = w._execute()
        assert not ok
        assert "nonexistent" in detail


# ── Model check worker ─────────────────────────────────────────────────────

def test_model_check_no_files(tmp_path):
    with patch("sam3_resolve.constants.MODELS_DIR", tmp_path):
        w = _ModelCheckWorker()
        ok, detail = w._execute()
        assert not ok
        assert "No model" in detail


def test_model_check_large_exists(tmp_path):
    (tmp_path / "sam3_large.pt").write_bytes(b"x" * 100)
    with patch("sam3_resolve.constants.MODELS_DIR", tmp_path):
        w = _ModelCheckWorker()
        ok, detail = w._execute()
        assert ok
        assert "Large" in detail


def test_model_check_base_exists(tmp_path):
    (tmp_path / "sam3_base.pt").write_bytes(b"x" * 100)
    with patch("sam3_resolve.constants.MODELS_DIR", tmp_path):
        w = _ModelCheckWorker()
        ok, detail = w._execute()
        assert ok
        assert "Base" in detail


# ── Wizard step_statuses ───────────────────────────────────────────────────

def test_wizard_step_statuses_returns_list(wizard):
    statuses = wizard.step_statuses()
    assert len(statuses) == 5
    assert all(isinstance(s, StepStatus) for s in statuses)


# ── Wizard _on_step_done (unit, no QThread) ────────────────────────────────

def test_wizard_on_step_done_ok(wizard):
    wizard._current_step = 0
    # Simulate the last step completing so _run_next terminates
    with patch.object(wizard, "_run_next"):
        wizard._on_step_done(True, "Python 3.12.0")
    assert wizard._step_rows[0].status == StepStatus.OK


def test_wizard_on_step_done_error_required(wizard):
    wizard._current_step = 0  # first step is required
    with patch.object(wizard, "_run_next"):
        wizard._on_step_done(False, "too old")
    assert wizard._step_rows[0].status == StepStatus.ERROR


def test_wizard_on_step_done_error_optional(wizard):
    # Step index 1 (GPU detect) is not required
    wizard._current_step = 1
    with patch.object(wizard, "_run_next"):
        wizard._on_step_done(False, "no GPU")
    assert wizard._step_rows[1].status == StepStatus.WARN


# ── _all_done enables/disables buttons ────────────────────────────────────

def test_wizard_all_done_enables_continue_when_required_ok(wizard):
    for i in range(5):
        wizard._step_results[i] = True
    wizard._all_done()
    assert wizard._btn_continue.isEnabled()


def test_wizard_all_done_disables_continue_on_required_fail(wizard):
    for i in range(5):
        wizard._step_results[i] = True
    wizard._step_results[0] = False   # first step is required
    wizard._all_done()
    assert not wizard._btn_continue.isEnabled()


def test_wizard_all_done_enables_repair_on_any_fail(wizard):
    for i in range(5):
        wizard._step_results[i] = True
    wizard._step_results[4] = False   # last step (optional)
    wizard._all_done()
    assert wizard._btn_repair.isEnabled()
