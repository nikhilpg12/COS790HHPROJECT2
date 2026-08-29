from __future__ import annotations

import json
import time
from typing import Any

from .protocol import LLMClientResponse


class OllamaClient:
    def __init__(self, model: str = "qwen3-coder:30b", temperature: float = 0.4) -> None:
        self.model = model
        self.temperature = temperature

    def complete_json(self, *, operation: str, prompt: str, seed: int) -> LLMClientResponse:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("The 'ollama' Python package is not installed. Run pip install -e .") from exc

        started = time.perf_counter()
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON matching the requested shape. Do not use markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"seed": int(seed), "temperature": float(self.temperature)},
        )
        latency = time.perf_counter() - started

        message = response["message"] if isinstance(response, dict) else response.message
        raw = message["content"] if isinstance(message, dict) else message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Ollama JSON response must be an object")

        def get_counter(name: str) -> int | None:
            if isinstance(response, dict):
                value = response.get(name)
            else:
                value = getattr(response, name, None)
            return int(value) if isinstance(value, (int, float)) else None

        return LLMClientResponse(
            data=data,
            raw_text=raw,
            latency_seconds=latency,
            prompt_tokens=get_counter("prompt_eval_count"),
            completion_tokens=get_counter("eval_count"),
        )
