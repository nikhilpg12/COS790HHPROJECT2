from __future__ import annotations

import json
import time

from .protocol import LLMClientResponse


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        temperature: float = 0.4,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def complete_json(
        self,
        *,
        operation: str,
        prompt: str,
        seed: int,
    ) -> LLMClientResponse:

        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Run pip install -e ."
            ) from exc

        started = time.perf_counter()

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching the requested shape. "
                        "Do not use markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format="json",
            options={
                "seed": int(seed),
                "temperature": float(self.temperature),
            },
        )

        latency = time.perf_counter() - started

        message = (
            response["message"]
            if isinstance(response, dict)
            else response.message
        )

        raw = (
            message["content"]
            if isinstance(message, dict)
            else message.content
        )

        def get_counter(name: str) -> int | None:
            if isinstance(response, dict):
                value = response.get(name)
            else:
                value = getattr(
                    response,
                    name,
                    None,
                )

            if isinstance(value, (int, float)):
                return int(value)

            return None

        prompt_tokens = get_counter(
            "prompt_eval_count"
        )

        completion_tokens = get_counter(
            "eval_count"
        )

        # =========================================================
        # JSON VALIDATION
        # =========================================================
        #
        # A malformed LLM response is NOT a fatal program error.
        #
        # We return the raw response and error to QwenTreeOperators.
        # The operator can then:
        #
        #     retry
        #       ↓
        #     retry
        #       ↓
        #     raise LLMGenerationError
        #
        # evolution.py can then skip the failed crossover/mutation.
        # =========================================================

        try:
            data = json.loads(raw)

        except json.JSONDecodeError as exc:

            return LLMClientResponse(
                data={},
                raw_text=raw,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=(
                    "Ollama returned invalid JSON: "
                    f"{exc}"
                ),
            )

        if not isinstance(data, dict):

            return LLMClientResponse(
                data={},
                raw_text=raw,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=(
                    "Ollama JSON response must be an object"
                ),
            )

        return LLMClientResponse(
            data=data,
            raw_text=raw,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=None,
        )