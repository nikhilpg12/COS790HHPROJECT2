from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class LLMClientResponse:
    data: Mapping[str, Any]
    raw_text: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    operation: str
    seed: int
    prompt: str
    raw_response: str
    latency_seconds: float
    valid: bool
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generation: int | None = None
    parent_ids: tuple[str, ...] = ()


class JSONLLMClient(Protocol):
    def complete_json(self, *, operation: str, prompt: str, seed: int) -> LLMClientResponse:
        ...


class LLMGenerationError(RuntimeError):
    """Raised when an LLM operator exhausts its automated retry budget."""
