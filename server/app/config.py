import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed for runtime
    load_dotenv = None

if load_dotenv:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    app_name: str = "QueueStorm Investigator"
    groq_api_key: str = _env_first("GROQ_API_KEY", "GroqApi", "Groq-Api", "Grok-Api")
    groq_model: str = _env_first("GROQ_MODEL", default="openai/gpt-oss-20b")
    use_llm: bool = _env_bool("USE_LLM")
    llm_timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", "4")
    confidence_threshold: float = _env_float("confidence", os.getenv("CONFIDENCE", "0.75"))
    llm_min_accept_confidence: float = _env_float("LLM_MIN_ACCEPT_CONFIDENCE", "0.55")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1800"))

    debug_log_enabled: bool = _env_bool("DEBUG_LOG_ENABLED")
    debug_log_to_console: bool = _env_bool("DEBUG_LOG_TO_CONSOLE", "true")
    debug_log_dir: str = os.getenv("DEBUG_LOG_DIR", "logs/requests")
    debug_log_file: str = os.getenv("DEBUG_LOG_FILE", "")
    debug_log_llm_prompt: bool = _env_bool("DEBUG_LOG_LLM_PROMPT", "true")
    debug_log_llm_output: bool = _env_bool("DEBUG_LOG_LLM_OUTPUT", "true")


settings = Settings()
