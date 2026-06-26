from typing import Any, Dict

from .config import settings


def extract_with_llm(_: str) -> Dict[str, Any]:
    """Optional Groq hook for future language extraction.

    The competition-critical path is deterministic. This function intentionally
    returns an empty result unless USE_LLM=true and a Groq key is configured.
    """
    if not settings.use_llm or not settings.groq_api_key:
        return {}
    try:
        from groq import Groq
    except ImportError:
        return {}

    # Kept deliberately conservative: no final decision is delegated to the LLM.
    # It can be expanded later to return internal extraction hints.
    _client = Groq(api_key=settings.groq_api_key)
    return {}

