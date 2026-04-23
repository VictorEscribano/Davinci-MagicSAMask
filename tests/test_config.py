"""Unit tests for Config singleton."""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from sam3_resolve.config import Config


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure each test starts with a fresh Config singleton."""
    Config.reset()
    yield
    Config.reset()


def _make_config(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    return cfg_file


def test_read_installed_flag(tmp_path):
    cfg_file = _make_config(tmp_path, {"installed": True})
    cfg = Config(path=cfg_file)
    assert cfg.installed is True


def test_write_and_reload(tmp_path):
    cfg_file = _make_config(tmp_path, {"installed": False})
    cfg = Config(path=cfg_file)
    cfg["installed"] = True
    cfg.save()
    cfg.reload()
    assert cfg["installed"] is True


def test_atomic_save_does_not_leave_tmp(tmp_path):
    cfg_file = _make_config(tmp_path, {})
    cfg = Config(path=cfg_file)
    cfg["x"] = 99
    cfg.save()
    assert not (tmp_path / "config.json.tmp").exists()
    assert cfg_file.exists()


def test_missing_file_returns_empty(tmp_path):
    cfg = Config(path=tmp_path / "nonexistent.json")
    assert cfg.get("anything") is None


def test_active_model_default(tmp_path):
    cfg = Config(path=tmp_path / "nonexistent.json")
    assert cfg.active_model == "sam3_large"


def test_singleton_identity():
    import sam3_resolve.constants as c
    c.CONFIG_PATH = pathlib.Path(tempfile.mktemp(suffix=".json"))
    a = Config.instance()
    b = Config.instance()
    assert a is b
