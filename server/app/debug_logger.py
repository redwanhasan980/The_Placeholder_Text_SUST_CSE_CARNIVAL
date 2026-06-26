import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings

try:
    import structlog
except ImportError:  # pragma: no cover - local fallback when dependency is not installed yet
    structlog = None


class DecisionLogger:
    """Request-scoped structured decision logger.

    When enabled, each analyzed ticket gets its own JSONL file under
    DEBUG_LOG_DIR. Each line is one clean decision-stage event.
    """

    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.enabled = settings.debug_log_enabled
        self.log_path: Path | None = None
        self._handler: logging.Handler | None = None
        self._base_logger: logging.Logger | None = None
        self._logger = None

        if not self.enabled:
            return

        self.request_log_id = _request_log_id(ticket_id)
        self.log_path = _request_log_path(self.request_log_id)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger_name = f"queuestorm.decision.{self.request_log_id}"
        self._base_logger = logging.getLogger(logger_name)
        self._base_logger.setLevel(logging.INFO)
        self._base_logger.propagate = False
        self._base_logger.handlers.clear()

        self._handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._base_logger.addHandler(self._handler)

        if settings.debug_log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._base_logger.addHandler(console_handler)

        if structlog:
            self._logger = structlog.wrap_logger(
                self._base_logger,
                processors=[
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.JSONRenderer(ensure_ascii=False, sort_keys=False),
                ],
            ).bind(ticket_id=ticket_id, request_log_id=self.request_log_id)

        self.step(
            "log_started",
            {
                "log_file": str(self.log_path),
                "logger": "structlog" if structlog else "python_logging_json_fallback",
            },
        )

    def step(self, name: str, data: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        clean_data = _json_safe(data or {})
        if self._logger is not None:
            self._logger.info(name, stage=name, data=clean_data)
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "info",
            "event": name,
            "stage": name,
            "ticket_id": self.ticket_id,
            "request_log_id": getattr(self, "request_log_id", None),
            "data": clean_data,
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        if self._base_logger:
            self._base_logger.info(line)

    def close(self) -> None:
        if not self.enabled or not self._base_logger:
            return
        for handler in list(self._base_logger.handlers):
            handler.flush()
            handler.close()
            self._base_logger.removeHandler(handler)


def _request_log_id(ticket_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_ticket = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticket_id).strip("._-") or "unknown_ticket"
    return f"{timestamp}_{safe_ticket}_{uuid.uuid4().hex[:8]}"


def _request_log_path(request_log_id: str) -> Path:
    base_dir = Path(settings.debug_log_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    return base_dir / f"{request_log_id}.jsonl"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if str(key).lower() not in {"groq_api_key", "api_key", "authorization"}
        }
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    if hasattr(value, "value"):
        return value.value
    return str(value)
