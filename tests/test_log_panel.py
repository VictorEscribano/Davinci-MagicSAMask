"""Tests for log_panel.LogPanel."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from sam3_resolve.ui.log_panel import LogPanel, MAX_LOG_LINES


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def panel(qt_app):
    return LogPanel()


# ── Basic append ───────────────────────────────────────────────────────────

def test_append_log_adds_line(panel):
    panel.clear_log()
    panel.append_log("INFO", "hello world")
    lines = panel.log_lines()
    assert any("hello world" in ln for ln in lines)


def test_append_log_level_present(panel):
    panel.clear_log()
    panel.append_log("OK", "model ready")
    assert any("[OK]" in ln for ln in panel.log_lines())


def test_append_multiple_lines(panel):
    panel.clear_log()
    for i in range(5):
        panel.append_log("INFO", f"line {i}")
    lines = panel.log_lines()
    assert len([ln for ln in lines if ln.strip()]) >= 5


# ── Level coverage ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["INFO", "OK", "WARN", "ERROR", "DEBUG"])
def test_known_levels_accepted(panel, level):
    panel.clear_log()
    panel.append_log(level, "test message")
    assert any("test message" in ln for ln in panel.log_lines())


def test_unknown_level_accepted(panel):
    panel.clear_log()
    panel.append_log("CUSTOM", "custom message")
    assert any("custom message" in ln for ln in panel.log_lines())


def test_level_case_insensitive(panel):
    panel.clear_log()
    panel.append_log("warn", "lower-case warn")
    assert any("lower-case warn" in ln for ln in panel.log_lines())


# ── Clear ──────────────────────────────────────────────────────────────────

def test_clear_log_empties_panel(panel):
    panel.append_log("INFO", "before clear")
    panel.clear_log()
    assert panel.log_lines() == [""] or panel.log_lines() == []


# ── Max block count ────────────────────────────────────────────────────────

def test_max_block_count_set(panel):
    assert panel.maximumBlockCount() == MAX_LOG_LINES


# ── Read-only ─────────────────────────────────────────────────────────────

def test_panel_is_readonly(panel):
    assert panel.isReadOnly()
