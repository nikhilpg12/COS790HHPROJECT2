from __future__ import annotations

from collections.abc import Mapping

from llm_gp_hh.gp.tree import Tree, parse_tree, tree_to_prefix, validate_tree

from .prompts import crossover_prompt, initial_prompt, mutation_prompt
from .protocol import JSONLLMClient, LLMCallRecord, LLMGenerationError


class QwenTreeOperators:
    def __init__(self, *, client: JSONLLMClient, retry_limit: int = 2) -> None:
        if retry_limit < 1:
            raise ValueError("retry_limit must be at least 1")
        self.client = client
        self.retry_limit = retry_limit

    @staticmethod
    def _extract_trees(data: Mapping[str, object], expected_count: int) -> list[str]:
        trees = data.get("trees")
        if not isinstance(trees, list) or not all(isinstance(item, str) for item in trees):
            raise ValueError("response must contain a string list named 'trees'")
        if len(trees) != expected_count:
            raise ValueError(f"expected exactly {expected_count} tree(s), received {len(trees)}")
        return list(trees)

    @staticmethod
    def _record(operation: str, seed: int, prompt: str, response, *, valid: bool, error: str | None) -> LLMCallRecord:
        return LLMCallRecord(
            operation=operation,
            seed=seed,
            prompt=prompt,
            raw_response=response.raw_text,
            latency_seconds=response.latency_seconds,
            valid=valid,
            error=error,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def generate_initial(self, *, count: int, max_depth: int, seed: int) -> tuple[list[Tree], list[LLMCallRecord]]:
        records: list[LLMCallRecord] = []
        error: str | None = None
        for attempt in range(self.retry_limit):
            call_seed = seed + attempt
            prompt = initial_prompt(count=count, max_depth=max_depth, error=error)
            response = self.client.complete_json(operation="initial", prompt=prompt, seed=call_seed)
            try:
                raw_trees = self._extract_trees(response.data, count)
                parsed = [parse_tree(text) for text in raw_trees]
                for tree in parsed:
                    validate_tree(tree, max_depth=max_depth)
                canonical = [tree_to_prefix(tree) for tree in parsed]
                if len(set(canonical)) != len(canonical):
                    raise ValueError("initial generation contains duplicate trees")
            except Exception as exc:
                error = str(exc)
                records.append(self._record("initial", call_seed, prompt, response, valid=False, error=error))
                continue
            records.append(self._record("initial", call_seed, prompt, response, valid=True, error=None))
            return parsed, records
        raise LLMGenerationError(f"initial generation failed after {self.retry_limit} attempts: {error}")

    def crossover(
        self,
        *,
        parent_a: str,
        parent_a_metrics: Mapping[str, float | int],
        parent_b: str,
        parent_b_metrics: Mapping[str, float | int],
        seed: int,
    ) -> tuple[tuple[Tree, Tree], list[LLMCallRecord]]:
        records: list[LLMCallRecord] = []
        error: str | None = None
        parent_set = {tree_to_prefix(parse_tree(parent_a)), tree_to_prefix(parse_tree(parent_b))}
        for attempt in range(self.retry_limit):
            call_seed = seed + attempt
            prompt = crossover_prompt(
                parent_a=parent_a,
                parent_a_metrics=parent_a_metrics,
                parent_b=parent_b,
                parent_b_metrics=parent_b_metrics,
                error=error,
            )
            response = self.client.complete_json(operation="crossover", prompt=prompt, seed=call_seed)
            try:
                raw_trees = self._extract_trees(response.data, 2)
                parsed = [parse_tree(text) for text in raw_trees]
                for tree in parsed:
                    validate_tree(tree)
                canonical = [tree_to_prefix(tree) for tree in parsed]
                if canonical[0] == canonical[1]:
                    raise ValueError("crossover children are identical")
                if any(child in parent_set for child in canonical):
                    raise ValueError("crossover returned an unchanged parent copy")
            except Exception as exc:
                error = str(exc)
                records.append(self._record("crossover", call_seed, prompt, response, valid=False, error=error))
                continue
            records.append(self._record("crossover", call_seed, prompt, response, valid=True, error=None))
            return (parsed[0], parsed[1]), records
        raise LLMGenerationError(f"crossover failed after {self.retry_limit} attempts: {error}")

    def mutate(
        self,
        *,
        parent: str,
        parent_metrics: Mapping[str, float | int],
        seed: int,
    ) -> tuple[Tree, list[LLMCallRecord]]:
        records: list[LLMCallRecord] = []
        error: str | None = None
        canonical_parent = tree_to_prefix(parse_tree(parent))
        for attempt in range(self.retry_limit):
            call_seed = seed + attempt
            prompt = mutation_prompt(parent=parent, parent_metrics=parent_metrics, error=error)
            response = self.client.complete_json(operation="mutation", prompt=prompt, seed=call_seed)
            try:
                raw_trees = self._extract_trees(response.data, 1)
                child = parse_tree(raw_trees[0])
                validate_tree(child)
                if tree_to_prefix(child) == canonical_parent:
                    raise ValueError("mutation returned an unchanged parent")
            except Exception as exc:
                error = str(exc)
                records.append(self._record("mutation", call_seed, prompt, response, valid=False, error=error))
                continue
            records.append(self._record("mutation", call_seed, prompt, response, valid=True, error=None))
            return child, records
        raise LLMGenerationError(f"mutation failed after {self.retry_limit} attempts: {error}")
