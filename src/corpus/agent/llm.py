from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from corpus.config import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_STRONG_MODEL,
    GEMINI_FAST_MODEL,
    GEMINI_STRONG_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_FAST_MODEL,
    OLLAMA_STRONG_MODEL,
    OPENAI_FAST_MODEL,
    OPENAI_STRONG_MODEL,
)


class LLMProvider(Protocol):
    """Chat models used by the agent nodes."""

    @property
    def fast(self) -> BaseChatModel:
        """Low-latency model for PLAN / GRADE / REWRITE nodes."""
        ...

    @property
    def strong(self) -> BaseChatModel:
        """High-quality model for GENERATE node."""
        ...


class _ModelPairProvider:
    def __init__(self, fast: BaseChatModel, strong: BaseChatModel) -> None:
        self._fast = fast
        self._strong = strong

    @property
    def fast(self) -> BaseChatModel:
        return self._fast

    @property
    def strong(self) -> BaseChatModel:
        return self._strong


class OpenAIProvider(_ModelPairProvider):
    """Requires OPENAI_API_KEY in env."""

    def __init__(self, temperature: float = 0.0) -> None:
        super().__init__(
            fast=ChatOpenAI(model=OPENAI_FAST_MODEL, temperature=temperature),
            strong=ChatOpenAI(model=OPENAI_STRONG_MODEL, temperature=temperature),
        )


class AnthropicProvider(_ModelPairProvider):
    """Requires ANTHROPIC_API_KEY in env."""

    def __init__(self, temperature: float = 0.0) -> None:
        super().__init__(
            fast=ChatAnthropic(model=ANTHROPIC_FAST_MODEL, temperature=temperature),
            strong=ChatAnthropic(model=ANTHROPIC_STRONG_MODEL, temperature=temperature),
        )


class GeminiProvider(_ModelPairProvider):
    """Requires GOOGLE_API_KEY in env."""

    def __init__(self, temperature: float = 0.0) -> None:
        super().__init__(
            fast=ChatGoogleGenerativeAI(model=GEMINI_FAST_MODEL, temperature=temperature),
            strong=ChatGoogleGenerativeAI(model=GEMINI_STRONG_MODEL, temperature=temperature),
        )


class OllamaProvider(_ModelPairProvider):
    """Local Ollama. No key required."""

    def __init__(self) -> None:
        super().__init__(
            fast=ChatOllama(model=OLLAMA_FAST_MODEL, base_url=OLLAMA_BASE_URL),
            strong=ChatOllama(model=OLLAMA_STRONG_MODEL, base_url=OLLAMA_BASE_URL),
        )


def default_provider() -> LLMProvider:
    """Select an LLM provider from CORPUS_LLM_PROVIDER or available API keys."""
    explicit = os.getenv("CORPUS_LLM_PROVIDER", "").lower()
    providers: dict[str, Callable[[], LLMProvider]] = {
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }

    if explicit:
        if explicit not in providers:
            names = ", ".join(providers)
            raise ValueError(f"Unknown CORPUS_LLM_PROVIDER={explicit!r}. Expected one of: {names}.")
        return providers[explicit]()

    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    return OllamaProvider()
