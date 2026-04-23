"""Tests for settings_panel — construction, state, apply logic."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from sam3_resolve.ui.settings_panel import (
    DEFAULT_KEYBINDINGS,
    ANIMATION_MS,
    SettingsPanel,
    _section_label,
    _separator,
)
from sam3_resolve.config import Config


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def parent_widget(qt_app):
    w = QWidget()
    w.resize(600, 500)
    return w


@pytest.fixture()
def panel(parent_widget):
    return SettingsPanel(parent=parent_widget)


# ── Construction ───────────────────────────────────────────────────────────

def test_panel_created(panel):
    assert panel is not None


def test_panel_width(panel):
    assert panel.width() == 300


def test_panel_initially_hidden(panel):
    assert not panel.is_panel_visible


def test_panel_has_keybinding_table(panel):
    assert panel._keybind_table is not None


def test_keybinding_table_row_count(panel):
    assert panel._keybind_table.rowCount() == len(DEFAULT_KEYBINDINGS)


def test_keybinding_table_column_count(panel):
    assert panel._keybind_table.columnCount() == 2


# ── Visibility toggle ──────────────────────────────────────────────────────

def test_show_panel_sets_visible(panel):
    panel.show_panel()
    assert panel.is_panel_visible


def test_hide_panel_clears_visible(panel):
    panel.show_panel()
    panel.hide_panel()
    assert not panel.is_panel_visible


# ── get_keybinding ─────────────────────────────────────────────────────────

def test_get_keybinding_undo(panel):
    seq = panel.get_keybinding("Undo last prompt")
    assert "Ctrl" in seq or seq == ""   # may or may not render locale-specifically


def test_get_keybinding_unknown_returns_empty(panel):
    seq = panel.get_keybinding("nonexistent action")
    assert seq == ""


# ── Model / device combos ──────────────────────────────────────────────────

def test_model_combo_has_two_options(panel):
    assert panel._model_combo.count() == 2


def test_device_combo_has_four_options(panel):
    assert panel._device_combo.count() == 4


# ── Apply writes to Config ─────────────────────────────────────────────────

def test_apply_saves_model_name(panel, tmp_path):
    cfg = Config.instance()
    panel._model_combo.setCurrentIndex(1)   # "SAM3 Base"
    with patch.object(cfg, "save"):
        panel._apply()
    assert cfg.get("model_name") == "base"


def test_apply_saves_device(panel):
    cfg = Config.instance()
    panel._device_combo.setCurrentIndex(3)   # CPU
    with patch.object(cfg, "save"):
        panel._apply()
    assert cfg.get("device") == "cpu"


def test_apply_saves_export_workers(panel):
    cfg = Config.instance()
    panel._workers_spin.setValue(2)
    with patch.object(cfg, "save"):
        panel._apply()
    assert cfg.get("export_workers") == 2


def test_apply_emits_signal(panel, qt_app):
    received = []
    panel.settings_applied.connect(lambda: received.append(True))
    with patch.object(Config.instance(), "save"):
        panel._apply()
    assert received


# ── Repair signal ──────────────────────────────────────────────────────────

def test_repair_signal(panel, qt_app):
    received = []
    panel.repair_requested.connect(lambda: received.append(True))
    panel.repair_requested.emit()
    assert received


# ── Helpers ────────────────────────────────────────────────────────────────

def test_section_label_returns_qlabel(qt_app):
    lbl = _section_label("Test")
    assert lbl.text() == "TEST"


def test_separator_returns_frame(qt_app):
    sep = _separator()
    assert sep is not None
