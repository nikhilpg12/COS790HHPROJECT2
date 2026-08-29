from __future__ import annotations

from dataclasses import dataclass

from .tree import Tree


@dataclass(frozen=True, slots=True)
class Individual:
    id: str
    tree: Tree
    generation: int
    operation: str
    parent_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class EvaluatedIndividual(Individual):
    hcv: int
    scv: float
    fitness: float
    evaluation_seconds: float
