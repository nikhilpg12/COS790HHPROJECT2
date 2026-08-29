from __future__ import annotations

from collections.abc import Mapping, Set

from .fitness import PROXIMITY_WEIGHTS
from .model import TorontoInstance


def feasible_periods(
    instance: TorontoInstance,
    exam_id: int,
    assignments: Mapping[int, int],
) -> tuple[int, ...]:
    blocked = {
        assignments[other]
        for other in instance.adjacency[exam_id]
        if other in assignments
    }
    return tuple(period for period in range(instance.periods) if period not in blocked)


def incremental_proximity_penalty(
    instance: TorontoInstance,
    exam_id: int,
    period: int,
    assignments: Mapping[int, int],
) -> float:
    total = 0.0
    for other in instance.adjacency[exam_id]:
        if other not in assignments:
            continue
        pair = (exam_id, other) if exam_id < other else (other, exam_id)
        common_students = instance.conflicts.get(pair, 0)
        distance = abs(period - assignments[other])
        total += PROXIMITY_WEIGHTS.get(distance, 0) * common_students
    return total


def terminal_values(
    instance: TorontoInstance,
    exam_id: int,
    assignments: Mapping[int, int],
    unallocated: Set[int],
) -> dict[str, float]:
    if exam_id not in instance.enrolments:
        raise KeyError(f"unknown exam {exam_id}")

    neighbours = instance.adjacency[exam_id]
    feasible = feasible_periods(instance, exam_id, assignments)
    allocated_neighbours = [other for other in neighbours if other in assignments]

    # The paper describes terminal a as a dynamic weighted-distance measure and
    # states that it is initially zero. For an unallocated exam, the executable
    # interpretation used here is the minimum incremental Toronto proximity cost
    # achievable over its currently feasible periods. This is explicit and logged
    # because the paper does not give more implementation detail for this terminal.
    if allocated_neighbours and feasible:
        a_value = min(
            incremental_proximity_penalty(instance, exam_id, period, assignments)
            for period in feasible
        )
    else:
        a_value = 0.0

    return {
        "a": float(a_value),
        "b": float(sum(1 for other in neighbours if other in unallocated)),
        "c": float(len(neighbours)),
        "d": float(instance.enrolments[exam_id]),
        "e": float(len(feasible)),
        "f": float(sum(1 for other in neighbours if other in assignments)),
        "g": float(instance.total_students),
        "h": float(instance.periods),
    }
