from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from llm_gp_hh.gp.tree import Tree, evaluate_tree, validate_tree

from .attributes import feasible_periods, incremental_proximity_penalty, terminal_values
from .fitness import proximity_cost
from .model import TorontoInstance


@dataclass(frozen=True, slots=True)
class ConstructionResult:
    assignments: Mapping[int, int]
    unallocated: tuple[int, ...]
    hcv: int
    scv: float


def _choose_event(
    instance: TorontoInstance,
    tree: Tree,
    active: set[int],
    assignments: dict[int, int],
    rng: random.Random,
) -> int:
    scores: dict[int, float] = {}
    for exam_id in active:
        values = terminal_values(instance, exam_id, assignments, active)
        scores[exam_id] = evaluate_tree(tree, values)
    best_score = max(scores.values())
    tied = sorted(exam_id for exam_id, score in scores.items() if score == best_score)
    return tied[0] if len(tied) == 1 else tied[rng.randrange(len(tied))]


def _choose_period(
    instance: TorontoInstance,
    exam_id: int,
    assignments: dict[int, int],
    rng: random.Random,
) -> int | None:
    options = feasible_periods(instance, exam_id, assignments)
    if not options:
        return None
    penalties = {
        period: incremental_proximity_penalty(instance, exam_id, period, assignments)
        for period in options
    }
    minimum = min(penalties.values())
    tied = sorted(period for period, penalty in penalties.items() if penalty == minimum)
    return tied[0] if len(tied) == 1 else tied[rng.randrange(len(tied))]


def construct_timetable(
    instance: TorontoInstance,
    heuristic_tree: Tree,
    rng: random.Random,
) -> ConstructionResult:
    validate_tree(heuristic_tree)
    active = set(instance.exam_ids)
    assignments: dict[int, int] = {}
    unallocated: list[int] = []
    hcv = 0

    while active:
        exam_id = _choose_event(instance, heuristic_tree, active, assignments, rng)
        period = _choose_period(instance, exam_id, assignments, rng)
        if period is None:
            hcv += 1
            unallocated.append(exam_id)
        else:
            assignments[exam_id] = period
        active.remove(exam_id)

    return ConstructionResult(
        assignments=dict(assignments),
        unallocated=tuple(sorted(unallocated)),
        hcv=hcv,
        scv=proximity_cost(instance, assignments),
    )
