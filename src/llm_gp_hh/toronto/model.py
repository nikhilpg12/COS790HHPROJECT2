from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class TorontoInstance:
    name: str
    exam_ids: tuple[int, ...]
    enrolments: Mapping[int, int]
    students: tuple[tuple[int, ...], ...]
    periods: int
    conflicts: Mapping[tuple[int, int], int]
    adjacency: Mapping[int, frozenset[int]]

    @property
    def total_students(self) -> int:
        return len(self.students)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        exam_ids: tuple[int, ...],
        enrolments: dict[int, int],
        students: tuple[tuple[int, ...], ...],
        periods: int,
        conflicts: dict[tuple[int, int], int],
        adjacency: dict[int, set[int]],
    ) -> "TorontoInstance":
        return cls(
            name=name,
            exam_ids=exam_ids,
            enrolments=MappingProxyType(dict(enrolments)),
            students=students,
            periods=periods,
            conflicts=MappingProxyType(dict(conflicts)),
            adjacency=MappingProxyType({k: frozenset(v) for k, v in adjacency.items()}),
        )
