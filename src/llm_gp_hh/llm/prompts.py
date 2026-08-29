from __future__ import annotations

from collections.abc import Mapping

TERMINAL_TEXT = """Toronto terminals:
a = minimum incremental Toronto proximity penalty over currently feasible periods against allocated conflicting exams (0 initially)
b = clashes with unallocated exams
c = total potential clashes
d = exam enrolment
e = currently feasible periods
f = clashes with allocated exams
g = total students
h = total periods"""

GRAMMAR_TEXT = """Allowed prefix-tree grammar:
NUMERIC := a|b|c|d|e|f|g|h
         | (+ NUMERIC NUMERIC)
         | (- NUMERIC NUMERIC)
         | (* NUMERIC NUMERIC)
         | (/ NUMERIC NUMERIC)
         | (if REL NUMERIC NUMERIC)
REL     := (< NUMERIC NUMERIC) | (> NUMERIC NUMERIC)
         | (<= NUMERIC NUMERIC) | (>= NUMERIC NUMERIC)
         | (== NUMERIC NUMERIC) | (!= NUMERIC NUMERIC)
Division is protected by the evaluator: denominator zero returns 1.
Do not output Python, markdown, prose, constants, new terminals, or new functions."""

OUTPUT_TEXT = 'Return JSON only in this shape: {"trees": ["PREFIX_TREE", ...]}.'


def initial_prompt(*, count: int, max_depth: int, error: str | None = None) -> str:
    prompt = f"""You generate GP individuals for an arithmetic generation construction hyper-heuristic for Toronto examination timetabling.
{TERMINAL_TEXT}

{GRAMMAR_TEXT}

Generate exactly {count} distinct valid numeric heuristic trees. Each tree must have depth at most {max_depth}. Prefer structural diversity and a mixture of static and dynamic terminals.
{OUTPUT_TEXT}
"""
    if error:
        prompt += f"\nThe previous response was invalid: {error}\nCorrect that exact problem and return a fresh JSON response.\n"
    return prompt


def _metric_text(metrics: Mapping[str, float | int]) -> str:
    return f"fitness={metrics['fitness']}, HCV={metrics['hcv']}, SCV={metrics['scv']}"


def crossover_prompt(
    *,
    parent_a: str,
    parent_a_metrics: Mapping[str, float | int],
    parent_b: str,
    parent_b_metrics: Mapping[str, float | int],
    error: str | None = None,
) -> str:
    prompt = f"""You perform semantic crossover for GP individuals in an arithmetic generation construction hyper-heuristic for Toronto examination timetabling.
{TERMINAL_TEXT}

{GRAMMAR_TEXT}

Parent A: {parent_a}
Parent A metrics: {_metric_text(parent_a_metrics)}
Parent B: {parent_b}
Parent B metrics: {_metric_text(parent_b_metrics)}

Generate exactly two valid child trees. Combine useful structural or semantic ideas from both parents. Neither child may be an unchanged copy of either parent. The two children must also differ from each other.
{OUTPUT_TEXT}
"""
    if error:
        prompt += f"\nThe previous response was invalid: {error}\nCorrect that exact problem and return a fresh JSON response.\n"
    return prompt


def mutation_prompt(
    *,
    parent: str,
    parent_metrics: Mapping[str, float | int],
    error: str | None = None,
) -> str:
    prompt = f"""You perform semantic mutation for a GP individual in an arithmetic generation construction hyper-heuristic for Toronto examination timetabling.
{TERMINAL_TEXT}

{GRAMMAR_TEXT}

Parent: {parent}
Parent metrics: {_metric_text(parent_metrics)}

Generate exactly one valid mutated tree. Make a meaningful but controlled structural change that could improve feasibility or timetable quality. The child must not be an unchanged copy of the parent.
{OUTPUT_TEXT}
"""
    if error:
        prompt += f"\nThe previous response was invalid: {error}\nCorrect that exact problem and return a fresh JSON response.\n"
    return prompt
