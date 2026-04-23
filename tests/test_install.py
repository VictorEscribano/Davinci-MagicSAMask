"""Unit tests for install.py helper functions."""

from __future__ import annotations

import hashlib
import pathlib
import tempfile

import pytest

# Import just the pure helpers — no subprocess calls
from install import _torch_index_url, _validate_checkpoint


def test_torch_index_cuda12():
    assert "cu12" in _torch_index_url("12.1")


def test_torch_index_cuda11():
    assert "cu118" in _torch_index_url("11.8")


def test_torch_index_unknown_defaults_latest():
    url = _torch_index_url("")
    assert "cu" in url


def test_validate_checkpoint_missing_file(tmp_path):
    assert _validate_checkpoint(tmp_path / "nope.pt", "", 0) is False


def test_validate_checkpoint_size_mismatch(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x" * 100)
    # expected_size far from actual
    assert _validate_checkpoint(f, "", 100_000_000) is False


def test_validate_checkpoint_no_sha_passes(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x" * 100)
    # expected_size = 100, no sha → should pass
    assert _validate_checkpoint(f, "", 100) is True


def test_validate_checkpoint_sha_mismatch(tmp_path):
    f = tmp_path / "model.pt"
    data = b"hello world"
    f.write_bytes(data)
    wrong_sha = "0" * 64
    assert _validate_checkpoint(f, wrong_sha, len(data)) is False


def test_validate_checkpoint_correct_sha(tmp_path):
    f = tmp_path / "model.pt"
    data = b"hello world"
    f.write_bytes(data)
    correct_sha = hashlib.sha256(data).hexdigest()
    assert _validate_checkpoint(f, correct_sha, len(data)) is True
