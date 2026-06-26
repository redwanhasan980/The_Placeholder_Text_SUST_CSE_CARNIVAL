import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


class DecisionLogger:
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.enabled = settings.debug_log_enabled
        self.events: list[dict[str, Any]] = []

    def step(self, name: str, data: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticket_id": self.ticket_id,
            "step": name,
            "data": _json_safe(data or {}),
        }
        self.events.append(event)
        line = json.dumps(event, ensure_ascii=False, default=str)
        if settings.debug_log_to_console:
            print(line, flush=True)
        if settings.debug_log_file:
            path = Path(settings.debug_log_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if str(key).lower() not in {"groq_api_key", "api_key"}}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    if hasattr(value, "value"):
        return value.value
    return str(value)

