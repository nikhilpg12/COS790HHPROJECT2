from __future__ import annotations

import json
import time

from .protocol import LLMClientResponse


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        temperature: float = 0.4,
        *,
        transport_retries: int = 4,
        transport_backoff_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        # Transport-level resilience only: transient failures talking to the
        # Ollama server (connection reset, refused, timed out) are retried here
        # with backoff before the call is reported as failed. This does not
        # touch sampling behaviour -- the seed, temperature, model and prompt
        # are identical on every retry -- so it is neutral with respect to the
        # temperature sweep.
        self.transport_retries = max(1, int(transport_retries))
        self.transport_backoff_seconds = float(transport_backoff_seconds)

    def _chat_with_retry(self, prompt: str, seed: int):
        import ollama

        last_exc: Exception | None = None
        for attempt in range(self.transport_retries):
            try:
                return ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON matching the requested "
                                "shape. Do not use markdown fences."
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
            except Exception as exc:  # noqa: BLE001 - transport layer, retry all
                last_exc = exc
                if attempt == self.transport_retries - 1:
                    break
                time.sleep(
                    self.transport_backoff_seconds * (2 ** attempt)
                )

        assert last_exc is not None
        raise last_exc

    def complete_json(
        self,
        *,
        operation: str,
        prompt: str,
        seed: int,
    ) -> LLMClientResponse:

        try:
            import ollama  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Run pip install -e ."
            ) from exc

        started = time.perf_counter()

        try:
            response = self._chat_with_retry(prompt, seed)
        except Exception as exc:  # noqa: BLE001 - transient transport failure
            # A malformed response is not fatal; neither is a dropped
            # connection. Report it the same way so QwenTreeOperators can
            # retry and then skip the operation instead of aborting the run.
            return LLMClientResponse(
                data={},
                raw_text="",
                latency_seconds=time.perf_counter() - started,
                prompt_tokens=None,
                completion_tokens=None,
                error=(
                    "Ollama transport error after "
                    f"{self.transport_retries} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ),
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