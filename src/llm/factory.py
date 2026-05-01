from __future__ import annotations

import os

from src.llm.offline import OfflineStubLLM
from src.llm.anthropic import AnthropicLLM
from src.llm.openai import OpenAILLM


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "offline").lower().strip()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    return OfflineStubLLM()
