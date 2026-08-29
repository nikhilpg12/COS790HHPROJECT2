from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

from .model import TorontoInstance


def _read_crs(path: Path) -> dict[int, int]:
    enrolments: dict[int, int] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid .crs line {line_number}: expected '<exam> <enrolment>'")
        try:
            exam_id, enrolment = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"invalid integer in .crs line {line_number}") from exc
        if exam_id in enrolments:
            raise ValueError(f"duplicate exam id {exam_id} in .crs")
        if enrolment < 0:
            raise ValueError(f"negative enrolment for exam {exam_id}")
        enrolments[exam_id] = enrolment
    if not enrolments:
        raise ValueError(".crs file contains no exams")
    return enrolments


def _read_stu(path: Path, known_exams: set[int]) -> tuple[tuple[int, ...], ...]:
    students: list[tuple[int, ...]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            exams = tuple(int(token) for token in line.split())
        except ValueError as exc:
            raise ValueError(f"invalid integer in .stu line {line_number}") from exc
        unknown = sorted(set(exams) - known_exams)
        if unknown:
            raise ValueError(f"unknown exam id(s) in .stu line {line_number}: {unknown}")
        if len(set(exams)) != len(exams):
            raise ValueError(f"duplicate exam id in .stu line {line_number}")
        students.append(exams)
    if not students:
        raise ValueError(".stu file contains no students")
    return tuple(students)


def load_toronto(crs_path: Path | str, stu_path: Path | str, periods: int) -> TorontoInstance:
    crs = Path(crs_path)
    stu = Path(stu_path)
    if periods <= 0:
        raise ValueError("periods must be positive")
    enrolments = _read_crs(crs)
    students = _read_stu(stu, set(enrolments))

    conflicts_counter: Counter[tuple[int, int]] = Counter()
    adjacency: dict[int, set[int]] = {exam_id: set() for exam_id in enrolments}
    for exams in students:
        for a, b in combinations(sorted(exams), 2):
            conflicts_counter[(a, b)] += 1
            adjacency[a].add(b)
            adjacency[b].add(a)

    return TorontoInstance.create(
        name=crs.stem.lower(),
        exam_ids=tuple(sorted(enrolments)),
        enrolments=enrolments,
        students=students,
        periods=periods,
        conflicts=dict(conflicts_counter),
        adjacency=adjacency,
    )
