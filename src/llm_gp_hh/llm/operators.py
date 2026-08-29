from __future__ import annotations

from collections.abc import Mapping, Sequence

from llm_gp_hh.gp.tree import (
    Tree,
    parse_tree,
    tree_to_prefix,
    validate_tree,
)

from .prompts import (
    crossover_prompt,
    initial_prompt,
    mutation_prompt,
)

from .protocol import (
    JSONLLMClient,
    LLMCallRecord,
    LLMGenerationError,
)


class QwenTreeOperators:

    def __init__(
        self,
        *,
        client: JSONLLMClient,
        retry_limit: int = 2,
    ) -> None:

        if retry_limit < 1:
            raise ValueError(
                "retry_limit must be at least 1"
            )

        self.client = client
        self.retry_limit = retry_limit


    @staticmethod
    def _extract_trees(
        data: Mapping[str, object],
        expected_count: int,
    ) -> list[str]:
        """
        Strict extraction used by crossover and mutation.
        """

        trees = data.get("trees")

        if (
            not isinstance(trees, list)
            or not all(
                isinstance(item, str)
                for item in trees
            )
        ):
            raise ValueError(
                "response must contain a string "
                "list named 'trees'"
            )

        if len(trees) != expected_count:
            raise ValueError(
                f"expected exactly {expected_count} "
                f"tree(s), received {len(trees)}"
            )

        return list(trees)


    @staticmethod
    def _extract_initial_trees(
        data: Mapping[str, object],
    ) -> list[str]:
        """
        Lenient extraction used only for initialisation.

        If Qwen returns fewer trees than requested, the valid trees can still
        be accepted and Qwen can be asked only for the remaining population.
        """

        trees = data.get("trees")

        if (
            not isinstance(trees, list)
            or not all(
                isinstance(item, str)
                for item in trees
            )
        ):
            raise ValueError(
                "response must contain a string "
                "list named 'trees'"
            )

        if not trees:
            raise ValueError(
                "initial generation returned no trees"
            )

        return list(trees)


    @staticmethod
    def _record(
        operation: str,
        seed: int,
        prompt: str,
        response,
        *,
        valid: bool,
        error: str | None,
    ) -> LLMCallRecord:

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


    def generate_initial(
        self,
        *,
        count: int,
        max_depth: int,
        seed: int,
        existing_trees: Sequence[Tree] = (),
    ) -> tuple[
        list[Tree],
        list[LLMCallRecord],
    ]:
        """
        Ask Qwen for up to ``count`` initial GP individuals.

        Generation 0 candidates are checked only for GP validity.
        Exact and structurally repeated trees are permitted.

        ``existing_trees`` is retained only for compatibility with callers;
        it is deliberately not used to filter or guide generation.
        """

        if count < 1:
            raise ValueError(
                "count must be at least 1"
            )

        records: list[LLMCallRecord] = []
        accepted: list[Tree] = []
        error: str | None = None

        for attempt in range(
            self.retry_limit
        ):
            remaining = count - len(accepted)

            if remaining <= 0:
                break

            call_seed = seed + attempt

            prompt = initial_prompt(
                count=remaining,
                max_depth=max_depth,
                error=error,
            )

            response = (
                self.client.complete_json(
                    operation="initial",
                    prompt=prompt,
                    seed=call_seed,
                )
            )

            if response.error:
                error = response.error

                records.append(
                    self._record(
                        "initial",
                        call_seed,
                        prompt,
                        response,
                        valid=False,
                        error=error,
                    )
                )

                continue

            try:
                raw_trees = (
                    self._extract_initial_trees(
                        response.data
                    )
                )

            except Exception as exc:
                error = str(exc)

                records.append(
                    self._record(
                        "initial",
                        call_seed,
                        prompt,
                        response,
                        valid=False,
                        error=error,
                    )
                )

                continue

            requested_this_call = remaining
            accepted_this_call = 0
            rejection_messages: list[str] = []

            for raw_tree in raw_trees:
                if len(accepted) >= count:
                    break

                try:
                    tree = parse_tree(raw_tree)

                    validate_tree(
                        tree,
                        max_depth=max_depth,
                    )

                except Exception as exc:
                    rejection_messages.append(
                        f"{raw_tree!r} -> {exc}"
                    )
                    continue

                accepted.append(tree)
                accepted_this_call += 1

            count_mismatch = (
                len(raw_trees)
                != requested_this_call
            )

            if (
                not rejection_messages
                and not count_mismatch
                and accepted_this_call
                == requested_this_call
            ):
                call_valid = True
                call_error = None
            else:
                call_valid = False

                parts: list[str] = []

                if count_mismatch:
                    parts.append(
                        "requested "
                        f"{requested_this_call} tree(s) "
                        f"but received {len(raw_trees)}"
                    )

                if rejection_messages:
                    parts.append(
                        "rejected "
                        f"{len(rejection_messages)} invalid tree(s): "
                        + " | ".join(
                            rejection_messages
                        )
                    )

                if accepted_this_call:
                    parts.append(
                        "partial acceptance: "
                        f"{accepted_this_call} valid tree(s) kept"
                    )
                else:
                    parts.append(
                        "no valid trees accepted"
                    )

                call_error = "; ".join(parts)

            records.append(
                self._record(
                    "initial",
                    call_seed,
                    prompt,
                    response,
                    valid=call_valid,
                    error=call_error,
                )
            )

            remaining = count - len(accepted)

            if remaining <= 0:
                break

            if rejection_messages:
                error = (
                    "Some generated trees were invalid. "
                    "Correct the reported validation errors. "
                    + " | ".join(
                        rejection_messages
                    )
                )
            elif count_mismatch:
                error = (
                    "The previous response did not "
                    "return the requested number of "
                    "trees. Return exactly "
                    f"{remaining} tree(s)."
                )
            else:
                error = (
                    "The previous response did not "
                    "supply enough valid trees. "
                    f"Return {remaining} tree(s)."
                )

        return accepted, records


    def crossover(
        self,
        *,
        parent_a: str,
        parent_a_metrics: Mapping[
            str,
            float | int,
        ],
        parent_b: str,
        parent_b_metrics: Mapping[
            str,
            float | int,
        ],
        seed: int,
    ) -> tuple[
        tuple[Tree, Tree],
        list[LLMCallRecord],
    ]:

        records: list[
            LLMCallRecord
        ] = []

        error: str | None = None

        parent_set = {
            tree_to_prefix(
                parse_tree(parent_a)
            ),
            tree_to_prefix(
                parse_tree(parent_b)
            ),
        }

        for attempt in range(
            self.retry_limit
        ):

            call_seed = seed + attempt

            prompt = crossover_prompt(
                parent_a=parent_a,
                parent_a_metrics=(
                    parent_a_metrics
                ),
                parent_b=parent_b,
                parent_b_metrics=(
                    parent_b_metrics
                ),
                error=error,
            )

            response = (
                self.client.complete_json(
                    operation="crossover",
                    prompt=prompt,
                    seed=call_seed,
                )
            )

            try:

                if response.error:
                    raise ValueError(
                        response.error
                    )

                raw_trees = (
                    self._extract_trees(
                        response.data,
                        2,
                    )
                )

                parsed = [
                    parse_tree(text)
                    for text in raw_trees
                ]

                for tree in parsed:
                    validate_tree(tree)

                canonical = [
                    tree_to_prefix(tree)
                    for tree in parsed
                ]

                if (
                    canonical[0]
                    == canonical[1]
                ):
                    raise ValueError(
                        "crossover children "
                        "are identical"
                    )

                if any(
                    child in parent_set
                    for child in canonical
                ):
                    raise ValueError(
                        "crossover returned "
                        "an unchanged parent copy"
                    )

            except Exception as exc:

                error = str(exc)

                records.append(
                    self._record(
                        "crossover",
                        call_seed,
                        prompt,
                        response,
                        valid=False,
                        error=error,
                    )
                )

                continue

            records.append(
                self._record(
                    "crossover",
                    call_seed,
                    prompt,
                    response,
                    valid=True,
                    error=None,
                )
            )

            return (
                parsed[0],
                parsed[1],
            ), records

        raise LLMGenerationError(
            "crossover failed "
            f"after {self.retry_limit} "
            f"attempts: {error}"
        )


    def mutate(
        self,
        *,
        parent: str,
        parent_metrics: Mapping[
            str,
            float | int,
        ],
        seed: int,
    ) -> tuple[
        Tree,
        list[LLMCallRecord],
    ]:

        records: list[
            LLMCallRecord
        ] = []

        error: str | None = None

        canonical_parent = (
            tree_to_prefix(
                parse_tree(parent)
            )
        )

        for attempt in range(
            self.retry_limit
        ):

            call_seed = seed + attempt

            prompt = mutation_prompt(
                parent=parent,
                parent_metrics=(
                    parent_metrics
                ),
                error=error,
            )

            response = (
                self.client.complete_json(
                    operation="mutation",
                    prompt=prompt,
                    seed=call_seed,
                )
            )

            try:

                if response.error:
                    raise ValueError(
                        response.error
                    )

                raw_trees = (
                    self._extract_trees(
                        response.data,
                        1,
                    )
                )

                child = parse_tree(
                    raw_trees[0]
                )

                validate_tree(child)

                if (
                    tree_to_prefix(child)
                    == canonical_parent
                ):
                    raise ValueError(
                        "mutation returned "
                        "an unchanged parent"
                    )

            except Exception as exc:

                error = str(exc)

                records.append(
                    self._record(
                        "mutation",
                        call_seed,
                        prompt,
                        response,
                        valid=False,
                        error=error,
                    )
                )

                continue

            records.append(
                self._record(
                    "mutation",
                    call_seed,
                    prompt,
                    response,
                    valid=True,
                    error=None,
                )
            )

            return child, records

        raise LLMGenerationError(
            "mutation failed "
            f"after {self.retry_limit} "
            f"attempts: {error}"
        )
