from __future__ import annotations

import random
from collections.abc import Sequence

from .individual import EvaluatedIndividual


def ranking_key(individual: EvaluatedIndividual) -> tuple[float, int, float, str]:
    """Rank individuals by Pillay & Ozcan product fitness, then deterministic ties."""
    return (individual.fitness, individual.hcv, individual.scv, individual.id)


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
    return min(sample, key=ranking_key)
