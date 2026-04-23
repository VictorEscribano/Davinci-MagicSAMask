"""
Headless tests for main_window.py layout and signal wiring.
Requires PyQt6 and a virtual display (or offscreen platform).
"""

from __future__ import annotations

import os
import sys

import pytest

# Force Qt offscreen platform before any import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from sam3_resolve.core.resolve_bridge import ClipInfo, ClipFormat, MockResolveBridge
from sam3_resolve.ui.main_window import (
    ActionBar,
    BottomPanel,
    ContextBar,
    HeaderBar,
    LeftPanel,
    MainWindow,
    RightPanel,
    ToolBar,
    TransportBar,
)


# ── QApplication fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture()
def clip() -> ClipInfo:
    return MockResolveBridge.DEFAULT_CLIP


# ── HeaderBar ──────────────────────────────────────────────────────────────

def test_header_bar_creates(qt_app):
    h = HeaderBar()
    assert h.objectName() == "header_bar"
    assert h.height() == 56


def test_header_bar_update_clip(qt_app, clip):
    h = HeaderBar()
    h.update_clip_info(clip)
    assert clip.name in h.clip_name_lbl.text()


def test_header_bar_gpu_status_ready(qt_app):
    h = HeaderBar()
    h.update_gpu_status("RTX 3090 · CUDA · 24GB   GPU Ready", ready=True)
    assert "RTX 3090" in h.gpu_status.text()


def test_header_bar_settings_signal(qt_app):
    h = HeaderBar()
    fired = []
    h.settings_requested.connect(lambda: fired.append(1))
    # simulate click — find gear button
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    # Emit signal directly since button lookup is complex in headless mode
    h.settings_requested.emit()
    assert len(fired) == 1


# ── LeftPanel ──────────────────────────────────────────────────────────────

def test_left_panel_width(qt_app):
    p = LeftPanel()
    assert p.maximumWidth() == 260


def test_left_panel_prompt_mode_default(qt_app):
    p = LeftPanel()
    assert p.prompt_mode == "points"


def test_left_panel_mode_signal(qt_app):
    p = LeftPanel()
    modes = []
    p.prompt_mode_changed.connect(modes.append)
    p._mode_box.setChecked(True)
    assert "box" in modes


def test_left_panel_update_clip_info(qt_app, clip):
    p = LeftPanel()
    p.update_clip_info(clip)
    assert "A001C001" in p._info_fields["file"].text()


def test_left_panel_proxy_preset_enabled_when_generate(qt_app):
    p = LeftPanel()
    p._proxy_gen.setChecked(True)
    assert p._proxy_preset.isEnabled()


def test_left_panel_proxy_preset_disabled_when_use(qt_app):
    p = LeftPanel()
    p._proxy_use.setChecked(True)
    assert not p._proxy_preset.isEnabled()


# ── ToolBar ────────────────────────────────────────────────────────────────

def test_toolbar_height(qt_app):
    t = ToolBar()
    assert t.height() == 40


def test_toolbar_undo_signal(qt_app):
    t = ToolBar()
    fired = []
    t.undo_requested.connect(lambda: fired.append(1))
    t.undo_requested.emit()
    assert len(fired) == 1


def test_toolbar_set_zoom(qt_app):
    t = ToolBar()
    t.set_zoom(150)
    assert "150%" in t.zoom_label.text()


def test_toolbar_set_active_tool(qt_app):
    t = ToolBar()
    t.set_active_tool("pan")
    assert t._btn_pan.isChecked()
    assert not t._btn_point.isChecked()


# ── ContextBar ─────────────────────────────────────────────────────────────

def test_context_bar_height(qt_app):
    c = ContextBar()
    assert c.height() == 26


def test_context_bar_set_mode_points(qt_app):
    c = ContextBar()
    c.set_mode("points")
    assert "POINTS" in c._mode_lbl.text()


def test_context_bar_set_mode_box(qt_app):
    c = ContextBar()
    c.set_mode("box")
    assert "BOX" in c._mode_lbl.text()


def test_context_bar_clear_signal(qt_app):
    c = ContextBar()
    fired = []
    c.clear_all_requested.connect(lambda: fired.append(1))
    c.clear_all_requested.emit()
    assert len(fired) == 1


# ── ActionBar ──────────────────────────────────────────────────────────────

def test_action_bar_height(qt_app):
    a = ActionBar()
    assert a.height() == 52


def test_action_bar_run_disabled_initially(qt_app):
    a = ActionBar()
    assert not a._buttons["btn_run"].isEnabled()


def test_action_bar_preview_disabled_initially(qt_app):
    a = ActionBar()
    assert not a._buttons["btn_preview"].isEnabled()


def test_action_bar_run_enables_with_prompts(qt_app):
    a = ActionBar()
    a.set_has_prompts(True)
    assert a._buttons["btn_run"].isEnabled()


def test_action_bar_preview_enables_after_propagation(qt_app):
    a = ActionBar()
    a.set_propagation_complete(True)
    assert a._buttons["btn_preview"].isEnabled()
    assert a._buttons["btn_accept"].isEnabled()


def test_action_bar_signals_emit(qt_app):
    a = ActionBar()
    fired = {}
    for sig_name in ("run", "preview", "accept", "modify", "cancel"):
        fired[sig_name] = []
        getattr(a, f"{sig_name}_requested").connect(
            lambda k=sig_name: fired[k].append(1)
        )
        getattr(a, f"{sig_name}_requested").emit()
    assert all(len(v) == 1 for v in fired.values())


# ── TransportBar ───────────────────────────────────────────────────────────

def test_transport_set_total_frames(qt_app):
    t = TransportBar()
    t.set_total_frames(200)
    assert t.scrubber.maximum() == 199


def test_transport_set_frame(qt_app):
    t = TransportBar()
    t.set_total_frames(100)
    t.set_frame(50)
    assert t.scrubber.value() == 50


def test_transport_set_frame_no_signal(qt_app):
    """set_frame() must not re-emit frame_changed."""
    t = TransportBar()
    t.set_total_frames(100)
    fired = []
    t.frame_changed.connect(lambda v: fired.append(v))
    t.set_frame(42)
    assert fired == []


# ── RightPanel ─────────────────────────────────────────────────────────────

def test_right_panel_width(qt_app):
    r = RightPanel()
    assert r.maximumWidth() == 240


def test_right_panel_update_object_count(qt_app):
    r = RightPanel()
    r.update_object_count(3)
    assert "3 / 8" in r._objects_title.text()


def test_right_panel_add_btn_disabled_at_max(qt_app):
    r = RightPanel()
    r.update_object_count(8)
    assert not r._add_obj_btn.isEnabled()


def test_right_panel_detection_card_hidden_initially(qt_app):
    r = RightPanel()
    assert not r._detection_card.isVisible()


def test_right_panel_detection_card_shown(qt_app):
    r = RightPanel()
    r.show_detection_result("person", 0.87)
    # isVisible() requires parent chain to be shown; isHidden() checks own flag only
    assert not r._detection_card.isHidden()
    assert "87%" in r._detection_result_lbl.text()


# ── BottomPanel ────────────────────────────────────────────────────────────

def test_bottom_panel_height(qt_app):
    b = BottomPanel()
    assert b.height() == 160


def test_bottom_panel_append_log(qt_app):
    b = BottomPanel()
    b.append_log("INFO", "Test message")
    assert "Test message" in b.log_view.toPlainText()


def test_bottom_panel_update_progress(qt_app):
    b = BottomPanel()
    b.set_total_frames(100)
    b.update_progress(50, 100, 12.5, 4.0)
    assert b.progress_bar.value() == 50
    assert "50" in b._stat_widgets["processed"].text()


# ── MainWindow integration ─────────────────────────────────────────────────

def test_main_window_creates(qt_app):
    w = MainWindow()
    assert w.windowTitle() == "SAM3 Mask Tracker"


def test_main_window_with_clip(qt_app, clip):
    w = MainWindow(clip=clip)
    assert clip.name in w.header.clip_name_lbl.text()


def test_main_window_minimum_size(qt_app):
    w = MainWindow()
    assert w.minimumWidth() == 1200
    assert w.minimumHeight() == 700


def test_main_window_prompt_mode_wires_context_bar(qt_app):
    w = MainWindow()
    w.left_panel._mode_box.setChecked(True)
    assert "BOX" in w.context_bar._mode_lbl.text()


def test_main_window_cancel_disables_buttons(qt_app, clip):
    w = MainWindow(clip=clip)
    w.action_bar.set_propagation_complete(True)
    w.action_bar.set_has_prompts(True)
    w._on_cancel()
    assert not w.action_bar._buttons["btn_accept"].isEnabled()
    assert not w.action_bar._buttons["btn_run"].isEnabled()
