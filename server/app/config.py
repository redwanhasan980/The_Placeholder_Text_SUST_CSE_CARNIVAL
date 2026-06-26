import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "QueueStorm Investigator"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    use_llm: bool = os.getenv("USE_LLM", "false").lower() in {"1", "true", "yes"}
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "4"))


settings = Settings()

