from __future__ import annotations

from collections.abc import Mapping

from .model import TorontoInstance

PROXIMITY_WEIGHTS = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}


def proximity_cost(instance: TorontoInstance, assignments: Mapping[int, int]) -> float:
    if instance.total_students <= 0:
        raise ValueError("Toronto instance must contain at least one student")
    total = 0.0
    for (exam_a, exam_b), common_students in instance.conflicts.items():
        if exam_a not in assignments or exam_b not in assignments:
            continue
        distance = abs(assignments[exam_a] - assignments[exam_b])
        total += PROXIMITY_WEIGHTS.get(distance, 0) * common_students
    return total / instance.total_students


def scalar_fitness(hcv: int, scv: float) -> float:
    if hcv < 0:
        raise ValueError("hcv cannot be negative")
    if scv < 0:
        raise ValueError("scv cannot be negative")
    return (hcv + 1) * float(scv)
