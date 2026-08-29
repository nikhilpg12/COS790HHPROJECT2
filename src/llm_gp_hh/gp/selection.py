from __future__ import annotations

import random
from collections.abc import Sequence

from .individual import EvaluatedIndividual


def tournament_select(
    population: Sequence[EvaluatedIndividual],
    tournament_size: int,
    rng: random.Random,
) -> EvaluatedIndividual:
    if not population:
        raise ValueError("population cannot be empty")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    sample = [population[rng.randrange(len(population))] for _ in range(tournament_size)]
    return min(sample, key=lambda item: (item.fitness, item.hcv, item.scv, item.id))
