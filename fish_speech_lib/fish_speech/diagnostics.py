# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

_DIAGNOSTIC_LOCK = threading.RLock()
_SESSION_ID = f"{os.getpid()}-{time.time_ns():x}"
_DEVICE_EVENTS: dict[str, dict[str, Any]] = {}
DIAGNOSTIC_LOG = Path(tempfile.gettempdir()) / f"fish_speech_devices_{os.getpid()}.json"


def diagnostic_log_path() -> Path:
    """Return the process's compact device snapshot in the system temp directory."""
    return DIAGNOSTIC_LOG


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def record_device_event(event: str, **fields: Any) -> None:
    """Replace each device's last setup event without appending logs or printing."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "fields": _json_safe(fields),
    }
    with _DIAGNOSTIC_LOCK:
        _DEVICE_EVENTS[str(fields.get("device", "default"))] = payload
        snapshot = {"session": _SESSION_ID, "pid": os.getpid(), "devices": _DEVICE_EVENTS}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=DIAGNOSTIC_LOG.parent,
                prefix="fish-speech-", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(snapshot, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temporary.replace(DIAGNOSTIC_LOG)
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
