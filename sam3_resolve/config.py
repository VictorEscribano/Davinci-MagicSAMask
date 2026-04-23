"""Singleton configuration manager backed by config.json."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from sam3_resolve.constants import CONFIG_PATH

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTANCE: "Config | None" = None


class Config:
    """
    Thread-safe singleton that reads/writes config.json.

    Usage:
        cfg = Config.instance()
        cfg["proxy"]["default_preset"] = "half"
        cfg.save()
    """

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self._path = path
        self._data: dict[str, Any] = {}
        self._load()

    # ── Singleton access ───────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "Config":
        """Return the process-wide Config singleton, creating it if needed."""
        global _INSTANCE
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = cls()
        return _INSTANCE

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (used in tests)."""
        global _INSTANCE
        with _LOCK:
            _INSTANCE = None

    # ── Public API ─────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def save(self) -> None:
        """
        Persist current state to disk atomically (write-then-rename).

        Raises:
            OSError: If the config directory is not writable.
        """
        tmp = self._path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            logger.debug("Config saved to %s", self._path)
        except OSError as exc:
            logger.error("Failed to save config: %s", exc)
            raise

    def reload(self) -> None:
        """Re-read config.json from disk, discarding any unsaved changes."""
        self._load()

    # ── Convenience helpers ────────────────────────────────────────────────

    @property
    def installed(self) -> bool:
        return bool(self._data.get("installed", False))

    @installed.setter
    def installed(self, value: bool) -> None:
        self._data["installed"] = value

    @property
    def active_model(self) -> str:
        return str(self._data.get("active_model", "sam3_large"))

    @active_model.setter
    def active_model(self, value: str) -> None:
        self._data["active_model"] = value

    @property
    def device(self) -> str:
        return str(self._data.get("device", "auto"))

    @device.setter
    def device(self, value: str) -> None:
        self._data["device"] = value

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.debug("Config loaded from %s", self._path)
            except json.JSONDecodeError as exc:
                logger.error("Corrupt config.json (%s); using defaults", exc)
                self._data = {}
        else:
            logger.warning("config.json not found at %s; using empty config", self._path)
            self._data = {}
