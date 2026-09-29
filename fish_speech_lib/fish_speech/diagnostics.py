from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from loguru import logger


_DIAGNOSTIC_LOCK = threading.RLock()
_SESSION_ID = f"{os.getpid()}-{time.time_ns():x}"
_DEFAULT_DIAGNOSTIC_LOG = Path(tempfile.gettempdir()) / "fish_speech_device_trace.log"
DIAGNOSTIC_LOG = Path(
    os.environ.get("FISH_SPEECH_DEVICE_LOG", str(_DEFAULT_DIAGNOSTIC_LOG))
).expanduser()


def diagnostic_log_path() -> Path:
    """Return the process-independent Fish device trace file path."""
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


def record_device_event(
    event: str,
    *,
    echo: bool = True,
    level: str = "INFO",
    **fields: Any,
) -> None:
    """Append one structured device event and optionally mirror it through Loguru.

    The file sink is intentionally independent of the host application's logging
    configuration so diagnostics survive when stderr/stdout are swallowed.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "session": _SESSION_ID,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "thread_id": threading.get_ident(),
        "event": event,
    }
    payload.update({key: _json_safe(value) for key, value in fields.items()})
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    try:
        with _DIAGNOSTIC_LOCK:
            DIAGNOSTIC_LOG.parent.mkdir(parents=True, exist_ok=True)
            with DIAGNOSTIC_LOG.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:
        if echo:
            logger.warning("Fish device trace file write failed: {}", exc)

    if echo:
        summary = " ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if key not in {"timestamp", "session", "pid", "thread_id", "event"}
        )
        logger.log(level, "Fish device trace [{}] {}", event, summary)
