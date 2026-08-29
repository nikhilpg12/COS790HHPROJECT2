from __future__ import annotations

from dataclasses import dataclass, replace
import random
import statistics
import time
from typing import Any

from llm_gp_hh.config import RunConfig
from llm_gp_hh.llm.protocol import LLMCallRecord, LLMGenerationError
from llm_gp_hh.toronto.constructor import construct_timetable
from llm_gp_hh.toronto.fitness import scalar_fitness
from llm_gp_hh.toronto.model import TorontoInstance

from .individual import EvaluatedIndividual
from .selection import tournament_select
from .tree import Tree, tree_to_prefix


@dataclass(frozen=True, slots=True)
class GenerationState:
    index: int
    population: tuple[EvaluatedIndividual, ...]
    operator_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    generations: tuple[GenerationState, ...]
    best: EvaluatedIndividual
    llm_calls: tuple[LLMCallRecord, ...]


def _metrics(individual: EvaluatedIndividual) -> dict[str, float | int]:
    return {
        "fitness": individual.fitness,
        "hcv": individual.hcv,
        "scv": individual.scv,
    }


def _say(message: str = "") -> None:
    """Print immediately so long-running local experiments never look frozen."""
    print(message, flush=True)


def _generation_summary(
    index: int,
    population: list[EvaluatedIndividual],
    operator_counts: dict[str, int],
) -> None:
    best = min(
        population,
        key=lambda item: (
            item.fitness,
            item.hcv,
            item.scv,
            item.id,
        ),
    )

    mean_fitness = statistics.fmean(
        item.fitness for item in population
    )

    mean_hcv = statistics.fmean(
        item.hcv for item in population
    )

    mean_scv = statistics.fmean(
        item.scv for item in population
    )

    _say(f"[GEN {index}] Complete")

    _say(
        f"        Best: {best.id} "
        f"| HCV={best.hcv} "
        f"| SCV={best.scv:.4f} "
        f"| Fitness={best.fitness:.4f}"
    )

    _say(
        f"        Mean: HCV={mean_hcv:.2f} "
        f"| SCV={mean_scv:.4f} "
        f"| Fitness={mean_fitness:.4f}"
    )

    if index > 0:
        _say(
            f"        Operators: "
            f"crossover={operator_counts['crossover']} "
            f"| mutation={operator_counts['mutation']}"
        )

    _say()


class EvolutionEngine:
    def _evaluate(
        self,
        *,
        instance: TorontoInstance,
        tree: Tree,
        identifier: str,
        generation: int,
        operation: str,
        parent_ids: tuple[str, ...],
        candidate_seed: int,
    ) -> EvaluatedIndividual:

        tree_text = tree_to_prefix(tree)

        _say(
            f"[EVAL] {identifier} "
            f"| evaluating {tree_text}"
        )

        started = time.perf_counter()

        construction = construct_timetable(
            instance,
            tree,
            random.Random(candidate_seed),
        )

        elapsed = time.perf_counter() - started

        fitness = scalar_fitness(
            construction.hcv,
            construction.scv,
        )

        result = EvaluatedIndividual(
            id=identifier,
            tree=tree,
            generation=generation,
            operation=operation,
            parent_ids=parent_ids,
            seed=candidate_seed,
            hcv=construction.hcv,
            scv=construction.scv,
            fitness=fitness,
            evaluation_seconds=elapsed,
        )

        _say(
            f"       HCV={result.hcv} "
            f"| SCV={result.scv:.4f} "
            f"| Fitness={result.fitness:.4f} "
            f"| {result.evaluation_seconds:.2f}s"
        )

        return result

    def run(
        self,
        instance: TorontoInstance,
        operators: Any,
        config: RunConfig,
        *,
        seed: int,
    ) -> EvolutionResult:

        rng = random.Random(seed)

        llm_calls: list[LLMCallRecord] = []
        generations: list[GenerationState] = []

        # =========================================================
        # RUN INFORMATION
        # =========================================================

        _say("=" * 72)

        _say(
            f"[RUN] {instance.name} "
            f"| model={config.model} "
            f"| seed={seed}"
        )

        _say(
            f"      population={config.population_size} "
            f"| generations={config.generations} "
            f"| tournament={config.tournament_size}"
        )

        _say(
            f"      crossover={config.crossover_rate:.2f} "
            f"| mutation={config.mutation_rate:.2f} "
            f"| initial_batch={config.initial_batch_size}"
        )

        _say("=" * 72)
        _say()

        # =========================================================
        # GENERATION 0
        # =========================================================

        initial: list[EvaluatedIndividual] = []

        _say(
            "[GEN 0] Creating initial population..."
        )

        while len(initial) < config.population_size:

            remaining = (
                config.population_size
                - len(initial)
            )

            count = min(
                config.initial_batch_size,
                remaining,
            )

            call_seed = rng.getrandbits(63)

            _say(
                f"[LLM] Requesting {count} "
                f"initial heuristic tree(s) from Qwen "
                f"({len(initial)}/"
                f"{config.population_size} created)..."
            )

            llm_started = time.perf_counter()

            trees, records = operators.generate_initial(
                count=count,
                max_depth=config.max_initial_depth,
                seed=call_seed,
            )

            llm_elapsed = (
                time.perf_counter()
                - llm_started
            )

            _say(
                f"[LLM] Initial generation returned "
                f"in {llm_elapsed:.2f}s"
            )

            llm_calls.extend(
                replace(
                    record,
                    generation=0,
                    parent_ids=(),
                )
                for record in records
            )

            if len(trees) != count:
                raise ValueError(
                    f"initial operator returned "
                    f"{len(trees)} trees; "
                    f"expected {count}"
                )

            for tree in trees:

                idx = len(initial)

                candidate_seed = (
                    rng.getrandbits(63)
                )

                initial.append(
                    self._evaluate(
                        instance=instance,
                        tree=tree,
                        identifier=f"g0-i{idx}",
                        generation=0,
                        operation="initial",
                        parent_ids=(),
                        candidate_seed=candidate_seed,
                    )
                )

        initial_state = GenerationState(
            index=0,
            population=tuple(initial),
            operator_counts={
                "crossover": 0,
                "mutation": 0,
            },
        )

        generations.append(
            initial_state
        )

        _generation_summary(
            0,
            initial,
            initial_state.operator_counts,
        )

        # =========================================================
        # EVOLUTION
        # =========================================================

        current = initial

        for generation_index in range(
            1,
            config.generations,
        ):

            next_population: list[
                EvaluatedIndividual
            ] = []

            operator_counts = {
                "crossover": 0,
                "mutation": 0,
            }

            _say(
                f"[GEN {generation_index}] "
                f"Starting generation..."
            )

            while (
                len(next_population)
                < config.population_size
            ):

                call_seed = (
                    rng.getrandbits(63)
                )

                # =============================================
                # CROSSOVER
                # =============================================

                if (
                    rng.random()
                    < config.crossover_rate
                ):

                    parent_a = tournament_select(
                        current,
                        config.tournament_size,
                        rng,
                    )

                    parent_b = tournament_select(
                        current,
                        config.tournament_size,
                        rng,
                    )

                    _say(
                        f"[SELECT] Crossover | "
                        f"{parent_a.id} "
                        f"(fit={parent_a.fitness:.4f}) "
                        f"+ "
                        f"{parent_b.id} "
                        f"(fit={parent_b.fitness:.4f})"
                    )

                    _say(
                        "[LLM] Asking Qwen "
                        "to perform crossover..."
                    )

                    llm_started = (
                        time.perf_counter()
                    )

                    try:
                        children, records = (
                            operators.crossover(
                                parent_a=(
                                    tree_to_prefix(
                                        parent_a.tree
                                    )
                                ),
                                parent_a_metrics=(
                                    _metrics(
                                        parent_a
                                    )
                                ),
                                parent_b=(
                                    tree_to_prefix(
                                        parent_b.tree
                                    )
                                ),
                                parent_b_metrics=(
                                    _metrics(
                                        parent_b
                                    )
                                ),
                                seed=call_seed,
                            )
                        )

                    except LLMGenerationError as exc:

                        llm_elapsed = (
                            time.perf_counter()
                            - llm_started
                        )

                        _say(
                            f"[SKIP] Crossover failed "
                            f"after {llm_elapsed:.2f}s: "
                            f"{exc}"
                        )

                        _say(
                            "       Skipping this "
                            "crossover attempt "
                            "and continuing..."
                        )

                        _say()

                        # Skip this failed crossover.
                        # The population slot is NOT lost.
                        # The loop selects another
                        # operator/parents and continues.
                        continue

                    llm_elapsed = (
                        time.perf_counter()
                        - llm_started
                    )

                    _say(
                        f"[LLM] Crossover returned "
                        f"in {llm_elapsed:.2f}s"
                    )

                    llm_calls.extend(
                        replace(
                            record,
                            generation=(
                                generation_index
                            ),
                            parent_ids=(
                                parent_a.id,
                                parent_b.id,
                            ),
                        )
                        for record in records
                    )

                    operator_counts[
                        "crossover"
                    ] += 1

                    for child in children:

                        if (
                            len(next_population)
                            >= config.population_size
                        ):
                            break

                        idx = len(
                            next_population
                        )

                        candidate_seed = (
                            rng.getrandbits(63)
                        )

                        next_population.append(
                            self._evaluate(
                                instance=instance,
                                tree=child,
                                identifier=(
                                    f"g{generation_index}"
                                    f"-i{idx}"
                                ),
                                generation=(
                                    generation_index
                                ),
                                operation=(
                                    "crossover"
                                ),
                                parent_ids=(
                                    parent_a.id,
                                    parent_b.id,
                                ),
                                candidate_seed=(
                                    candidate_seed
                                ),
                            )
                        )

                # =============================================
                # MUTATION
                # =============================================

                else:

                    parent = tournament_select(
                        current,
                        config.tournament_size,
                        rng,
                    )

                    _say(
                        f"[SELECT] Mutation | "
                        f"{parent.id} "
                        f"(HCV={parent.hcv}, "
                        f"SCV={parent.scv:.4f}, "
                        f"fit={parent.fitness:.4f})"
                    )

                    _say(
                        "[LLM] Asking Qwen "
                        "to mutate the parent..."
                    )

                    llm_started = (
                        time.perf_counter()
                    )

                    try:
                        child, records = (
                            operators.mutate(
                                parent=(
                                    tree_to_prefix(
                                        parent.tree
                                    )
                                ),
                                parent_metrics=(
                                    _metrics(
                                        parent
                                    )
                                ),
                                seed=call_seed,
                            )
                        )

                    except LLMGenerationError as exc:

                        llm_elapsed = (
                            time.perf_counter()
                            - llm_started
                        )

                        _say(
                            f"[SKIP] Mutation failed "
                            f"after {llm_elapsed:.2f}s: "
                            f"{exc}"
                        )

                        _say(
                            "       Skipping this "
                            "mutation attempt "
                            "and continuing..."
                        )

                        _say()

                        # Skip this failed mutation.
                        # The population slot is NOT lost.
                        # The loop simply tries again.
                        continue

                    llm_elapsed = (
                        time.perf_counter()
                        - llm_started
                    )

                    _say(
                        f"[LLM] Mutation returned "
                        f"in {llm_elapsed:.2f}s"
                    )

                    llm_calls.extend(
                        replace(
                            record,
                            generation=(
                                generation_index
                            ),
                            parent_ids=(
                                parent.id,
                            ),
                        )
                        for record in records
                    )

                    operator_counts[
                        "mutation"
                    ] += 1

                    idx = len(
                        next_population
                    )

                    candidate_seed = (
                        rng.getrandbits(63)
                    )

                    next_population.append(
                        self._evaluate(
                            instance=instance,
                            tree=child,
                            identifier=(
                                f"g{generation_index}"
                                f"-i{idx}"
                            ),
                            generation=(
                                generation_index
                            ),
                            operation="mutation",
                            parent_ids=(
                                parent.id,
                            ),
                            candidate_seed=(
                                candidate_seed
                            ),
                        )
                    )

                # =============================================
                # PROGRESS
                # =============================================

                _say(
                    f"[PROGRESS] Generation "
                    f"{generation_index}: "
                    f"{len(next_population)}/"
                    f"{config.population_size} "
                    f"offspring evaluated"
                )

                _say()

            # =================================================
            # GENERATION COMPLETE
            # =================================================

            state = GenerationState(
                index=generation_index,
                population=tuple(
                    next_population
                ),
                operator_counts=(
                    operator_counts
                ),
            )

            generations.append(
                state
            )

            _generation_summary(
                generation_index,
                next_population,
                operator_counts,
            )

            current = next_population

        # =========================================================
        # BEST OVERALL
        # =========================================================

        all_individuals = [
            item
            for state in generations
            for item in state.population
        ]

        best = min(
            all_individuals,
            key=lambda item: (
                item.fitness,
                item.hcv,
                item.scv,
                item.id,
            ),
        )

        _say("=" * 72)

        _say(
            f"[DONE] Best overall: "
            f"{best.id} "
            f"from generation "
            f"{best.generation}"
        )

        _say(
            f"       Tree: "
            f"{tree_to_prefix(best.tree)}"
        )

        _say(
            f"       HCV={best.hcv} "
            f"| SCV={best.scv:.4f} "
            f"| Fitness={best.fitness:.4f}"
        )

        _say("=" * 72)

        return EvolutionResult(
            generations=tuple(
                generations
            ),
            best=best,
            llm_calls=tuple(
                llm_calls
            ),
        )