"""
Tests for the Resolve API path discovery logic in resolve_bridge.py.

Covers:
  - RESOLVE_INSTALL_DIR env var (priority 1)
  - resolve_api_path in config.json (priority 2)
  - Standard OS paths (priority 3)
  - Glob fallback for non-standard installs (priority 4)
  - Path saved back to config after discovery
  - _inject_resolve_api_path adds path to sys.path

NOTE: Config is imported *locally* inside _candidate_resolve_api_paths() and
_inject_resolve_api_path(), so we patch "sam3_resolve.config.Config" — where
the class itself lives — not the resolve_bridge module namespace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sam3_resolve.core.resolve_bridge import (
    _candidate_resolve_api_paths,
    _inject_resolve_api_path,
)


def _mock_cfg(stored_path: str = "") -> MagicMock:
    """Helper: build a Config mock that returns stored_path for resolve_api_path."""
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: stored_path if k == "resolve_api_path" else d
    return cfg


# ── _candidate_resolve_api_paths ───────────────────────────────────────────

def test_env_var_appears_first(tmp_path, monkeypatch):
    """RESOLVE_INSTALL_DIR candidates come before everything else."""
    monkeypatch.setenv("RESOLVE_INSTALL_DIR", str(tmp_path))
    with patch("sam3_resolve.config.Config") as MockCfg:
        MockCfg.instance.return_value = _mock_cfg()
        candidates = _candidate_resolve_api_paths()
    assert str(tmp_path) in str(candidates[0])


def test_env_var_not_set_skipped(monkeypatch):
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    with patch("sam3_resolve.config.Config") as MockCfg:
        MockCfg.instance.return_value = _mock_cfg()
        candidates = _candidate_resolve_api_paths()
    assert all(str(c) for c in candidates)


def test_config_path_included(tmp_path, monkeypatch):
    """resolve_api_path from config.json appears in candidates."""
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    fake_path = str(tmp_path / "custom" / "Fusion")
    with patch("sam3_resolve.config.Config") as MockCfg:
        MockCfg.instance.return_value = _mock_cfg(fake_path)
        candidates = _candidate_resolve_api_paths()
    assert any(str(c) == fake_path for c in candidates)


def test_config_path_empty_not_added(monkeypatch):
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    with patch("sam3_resolve.config.Config") as MockCfg:
        MockCfg.instance.return_value = _mock_cfg("")
        candidates = _candidate_resolve_api_paths()
    assert not any(str(c) == "" for c in candidates)


def test_standard_linux_paths_included(monkeypatch):
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    with patch("platform.system", return_value="Linux"):
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = _mock_cfg()
            candidates = _candidate_resolve_api_paths()
    assert any("/opt/resolve" in str(c) for c in candidates)


def test_standard_darwin_paths_included(monkeypatch):
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    with patch("platform.system", return_value="Darwin"):
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = _mock_cfg()
            candidates = _candidate_resolve_api_paths()
    assert any("Applications" in str(c) for c in candidates)


def test_standard_windows_paths_included(monkeypatch):
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    with patch("platform.system", return_value="Windows"):
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = _mock_cfg()
            candidates = _candidate_resolve_api_paths()
    assert any("Blackmagic" in str(c) for c in candidates)


# ── _inject_resolve_api_path ───────────────────────────────────────────────

def test_inject_adds_existing_path_to_sys_path(tmp_path, monkeypatch):
    """If a candidate directory exists, it gets prepended to sys.path."""
    monkeypatch.setenv("RESOLVE_INSTALL_DIR", str(tmp_path))

    mock_cfg = _mock_cfg("")   # nothing stored yet
    original = sys.path[:]
    try:
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = mock_cfg
            _inject_resolve_api_path()
        assert str(tmp_path) in sys.path
    finally:
        sys.path[:] = original


def test_inject_saves_found_path_to_config(tmp_path, monkeypatch):
    """After finding a valid path, it is written to config.json."""
    monkeypatch.setenv("RESOLVE_INSTALL_DIR", str(tmp_path))

    mock_cfg = _mock_cfg("")   # nothing stored yet
    original = sys.path[:]
    try:
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = mock_cfg
            _inject_resolve_api_path()
    finally:
        sys.path[:] = original

    set_calls = {call[0][0]: call[0][1] for call in mock_cfg.set.call_args_list}
    assert "resolve_api_path" in set_calls
    assert str(tmp_path) in set_calls["resolve_api_path"]
    mock_cfg.save.assert_called_once()


def test_inject_skips_save_if_already_stored(tmp_path, monkeypatch):
    """If config already has a path, set() is NOT called for resolve_api_path."""
    monkeypatch.setenv("RESOLVE_INSTALL_DIR", str(tmp_path))

    mock_cfg = _mock_cfg(str(tmp_path))  # already stored
    original = sys.path[:]
    try:
        with patch("sam3_resolve.config.Config") as MockCfg:
            MockCfg.instance.return_value = mock_cfg
            _inject_resolve_api_path()
    finally:
        sys.path[:] = original

    set_keys = {call[0][0] for call in mock_cfg.set.call_args_list}
    assert "resolve_api_path" not in set_keys


def test_inject_nonexistent_candidates_dont_touch_sys_path(monkeypatch):
    """If no candidate exists, sys.path is unchanged."""
    monkeypatch.delenv("RESOLVE_INSTALL_DIR", raising=False)
    nonexistent = Path("/nonexistent/resolve/path/xyz123abc")

    original = sys.path[:]
    try:
        with patch(
            "sam3_resolve.core.resolve_bridge._candidate_resolve_api_paths",
            return_value=[nonexistent],
        ):
            _inject_resolve_api_path()
        new_entries = [p for p in sys.path if p not in original]
        assert not new_entries
    finally:
        sys.path[:] = original


# ── create_bridge fallback ─────────────────────────────────────────────────

def test_create_bridge_force_mock():
    from sam3_resolve.core.resolve_bridge import create_bridge, MockResolveBridge
    assert isinstance(create_bridge(force_mock=True), MockResolveBridge)


def test_create_bridge_falls_back_on_any_exception():
    """Any exception from RealResolveBridge → MockResolveBridge returned."""
    from sam3_resolve.core.resolve_bridge import create_bridge, MockResolveBridge
    with patch(
        "sam3_resolve.core.resolve_bridge.RealResolveBridge",
        side_effect=RuntimeError("Resolve not running"),
    ):
        bridge = create_bridge()
    assert isinstance(bridge, MockResolveBridge)


def test_create_bridge_falls_back_on_import_error():
    from sam3_resolve.core.resolve_bridge import create_bridge, MockResolveBridge
    with patch(
        "sam3_resolve.core.resolve_bridge.RealResolveBridge",
        side_effect=ImportError("DaVinciResolveScript not found"),
    ):
        bridge = create_bridge()
    assert isinstance(bridge, MockResolveBridge)
