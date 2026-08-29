from __future__ import annotations

from collections.abc import Mapping, Sequence


TERMINAL_SET_TEXT = """GP terminal set T:

a = minimum incremental Toronto proximity penalty over currently feasible periods against allocated conflicting exams (0 initially)
b = clashes with unallocated exams
c = total potential clashes
d = exam enrolment
e = currently feasible periods
f = clashes with allocated exams
g = total students
h = total periods
"""


FUNCTION_SET_TEXT = """GP function set F:

Arithmetic functions:
+  arity 2
-  arity 2
*  arity 2
/  arity 2

Relational functions:
<   arity 2
>   arity 2
<=  arity 2
>=  arity 2
==  arity 2
!=  arity 2

Conditional function:
if  arity 3

Division is protected:
if the denominator is zero, the evaluator returns 1.
"""


TREE_FORMAT_TEXT = """GP tree representation rules:

The GP individuals are expression trees.

Return each tree as a prefix textual representation of that tree.

A terminal is represented only by its terminal name:

TERMINAL

An arithmetic function node is represented as:

(ARITHMETIC_FUNCTION LEFT_NUMERIC_SUBTREE RIGHT_NUMERIC_SUBTREE)

A relational function node is represented as:

(RELATIONAL_FUNCTION LEFT_NUMERIC_SUBTREE RIGHT_NUMERIC_SUBTREE)

A conditional function node is represented as:

(if RELATIONAL_SUBTREE THEN_NUMERIC_SUBTREE ELSE_NUMERIC_SUBTREE)

Important structural rules:

- every arithmetic function has exactly two numeric child subtrees
- every relational function has exactly two numeric child subtrees
- a relational subtree may only appear as the condition of an if node
- every if node has exactly one relational condition, one numeric then subtree, and one numeric else subtree
- the root of the GP heuristic must return a numeric value
- every function expression begins with '(' and ends with ')'
- never output the word 'if' by itself
- never output an operator without its required child subtrees
- never introduce numeric constants
- use only terminals a through h
- use only functions from F
- do not introduce additional terminals or functions
"""


OUTPUT_TEXT = (
    'Return JSON only in this shape: '
    '{"trees": ["PREFIX_TREE", ...]}.'
)


def _accepted_population_text(
    accepted_population: Sequence[
        Mapping[str, str | int]
    ],
    existing_structures: Sequence[str],
) -> str:
    if accepted_population:
        lines: list[str] = [
            "CURRENT ACCEPTED INITIAL POPULATION ARCHIVE",
            "",
            (
                f"There are {len(accepted_population)} "
                "LLM-generated GP individuals already accepted."
            ),
            (
                "Use this archive only as diversity context. "
                "Exact or structural duplicates are allowed and will not be rejected solely for duplication."
            ),
            "",
        ]

        for index, entry in enumerate(
            accepted_population,
            start=1,
        ):
            lines.extend(
                [
                    f"Accepted individual {index}:",
                    f"Tree: {entry['tree']}",
                    f"Structural signature: {entry['structure']}",
                    f"Depth: {entry['depth']}",
                    "",
                ]
            )

        lines.extend(
            [
                "Diversity guidance:",
                "- aim for a varied set of valid trees when practical",
                "- exact or structural duplicates are allowed",
                "- duplication is not a validity failure",
            ]
        )

        return "\n".join(lines)

    if existing_structures:
        lines = "\n".join(
            f"- {structure}"
            for structure in existing_structures
        )

        return f"""CURRENT ACCEPTED STRUCTURAL SIGNATURES

{lines}

Use these only as diversity context. Duplicates are allowed.
"""

    return """CURRENT ACCEPTED INITIAL POPULATION ARCHIVE

No GP individuals have been accepted yet.
Create the first valid candidate set and aim for diversity where practical.
"""


def initial_prompt(
    *,
    count: int,
    max_depth: int,
    accepted_population: Sequence[
        Mapping[str, str | int]
    ] = (),
    needed_count: int | None = None,
    existing_structures: Sequence[str] = (),
    error: str | None = None,
) -> str:
    """
    Build the Generation 0 prompt.

    ``count`` is the number of candidate alternatives requested from Qwen.
    ``needed_count`` is the number of population slots still needing to be
    filled. Keeping them separate allows oversampling near the end of the
    population without allowing Python to generate or modify heuristics.
    """
    if count < 1:
        raise ValueError(
            "count must be at least 1"
        )

    if needed_count is None:
        needed_count = count

    if needed_count < 1:
        raise ValueError(
            "needed_count must be at least 1"
        )

    archive_text = _accepted_population_text(
        accepted_population,
        existing_structures,
    )

    prompt = f"""You generate individuals for a Genetic Programming generation construction hyper-heuristic for Toronto examination timetabling.

You replace the conventional random GP initialisation operator.

The LLM is the sole mechanism that creates GP individuals.
The host program only parses, validates, accepts, or rejects your candidates.
It does not create replacement heuristics.

You generate GP heuristic expression trees.
You do NOT generate timetables directly.

{TERMINAL_SET_TEXT}

{FUNCTION_SET_TEXT}

{TREE_FORMAT_TEXT}

{archive_text}

The initial population currently needs {needed_count} more accepted individual(s).

Generate exactly {count} candidate GP individuals.

When the requested candidate count is larger than the number of remaining population slots, the extra candidates are deliberate alternatives. Generate all requested candidates so that the host program has multiple choices and can accept only as many as are still needed.

Requirements:

- every candidate must be a valid GP expression tree
- use only terminals from T
- use only functions from F
- respect every function's arity
- each tree must have depth at most {max_depth}
- all candidates in this response must be exactly different from each other
- all candidates in this response must have different operator-aware structural signatures
- every candidate must be exactly different from every accepted tree in the archive
- every candidate must have a structural signature different from every accepted structural signature in the archive
- do not merely rename terminals in an existing tree structure
- vary tree topology
- vary root functions
- vary arithmetic operators
- vary relational operators where conditionals are used
- vary conditional placement
- vary subtree composition and depth
- vary terminal placement
- use both static and dynamic timetable attributes where useful
- do not output numeric constants
- do not output Python
- do not output Markdown
- do not output explanations or prose
- return only the requested JSON object

{OUTPUT_TEXT}
"""

    if error:
        prompt += (
            "\nThe previous candidate response had the following "
            "validation problem(s):\n"
            f"{error}\n\n"
            "Use that feedback as negative guidance. "
            "Generate fresh candidates that avoid the rejected exact "
            "trees and structures.\n"
        )

    return prompt


def _metric_text(
    metrics: Mapping[
        str,
        float | int,
    ],
) -> str:
    return (
        f"fitness={metrics['fitness']}, "
        f"HCV={metrics['hcv']}, "
        f"SCV={metrics['scv']}"
    )


def crossover_prompt(
    *,
    parent_a: str,
    parent_a_metrics: Mapping[
        str,
        float | int,
    ],
    parent_b: str,
    parent_b_metrics: Mapping[
        str,
        float | int,
    ],
    error: str | None = None,
) -> str:
    prompt = f"""You perform LLM-guided crossover on two Genetic Programming individuals in a generation construction hyper-heuristic for Toronto examination timetabling.

You replace the conventional GP crossover operator.

The parents and offspring are GP expression trees.
You do NOT generate timetables directly.

{TERMINAL_SET_TEXT}

{FUNCTION_SET_TEXT}

{TREE_FORMAT_TEXT}

Parent A:
{parent_a}

Parent A performance:
{_metric_text(parent_a_metrics)}

Parent B:
{parent_b}

Parent B performance:
{_metric_text(parent_b_metrics)}

Generate exactly two offspring GP expression trees.

Requirements:

- construct offspring using useful structures or relationships from the parents
- each child must be a valid GP expression tree
- use only terminals from T
- use only functions from F
- respect every function's arity
- neither child may be identical to Parent A
- neither child may be identical to Parent B
- the two offspring must differ from each other
- do not introduce numeric constants
- use the parent performance information as guidance
- use the Pillay and Ozcan search objective Fitness=(HCV+1)*SCV; lower fitness is better
- HCV=0 is the feasibility goal, but infeasible individuals are allowed during evolution
- among feasible solutions, lower SCV gives lower fitness and is therefore better
- do not output Python
- do not output Markdown
- do not output explanations or prose
- return only the requested JSON object

{OUTPUT_TEXT}
"""

    if error:
        prompt += (
            "\nThe previous response was invalid for this reason:\n"
            f"{error}\n\n"
            "Generate completely fresh offspring that correct that "
            "exact problem.\n"
        )

    return prompt


def mutation_prompt(
    *,
    parent: str,
    parent_metrics: Mapping[
        str,
        float | int,
    ],
    error: str | None = None,
) -> str:
    prompt = f"""You perform LLM-guided mutation on a Genetic Programming individual in a generation construction hyper-heuristic for Toronto examination timetabling.

You replace the conventional GP mutation operator.

The parent and offspring are GP expression trees.
You do NOT generate a timetable directly.

{TERMINAL_SET_TEXT}

{FUNCTION_SET_TEXT}

{TREE_FORMAT_TEXT}

Parent:
{parent}

Parent performance:
{_metric_text(parent_metrics)}

Generate exactly one mutated GP expression tree.

Requirements:

- make a meaningful but controlled modification to the parent tree
- the child must be a valid GP expression tree
- use only terminals from T
- use only functions from F
- respect every function's arity
- the child must differ from the parent
- retain useful parent structure where appropriate
- do not introduce numeric constants
- use the parent's performance information as guidance
- use the Pillay and Ozcan search objective Fitness=(HCV+1)*SCV; lower fitness is better
- HCV=0 is the feasibility goal, but infeasible individuals are allowed during evolution
- among feasible solutions, lower SCV gives lower fitness and is therefore better
- do not output Python
- do not output Markdown
- do not output explanations or prose
- return only the requested JSON object

{OUTPUT_TEXT}
"""

    if error:
        prompt += (
            "\nThe previous response was invalid for this reason:\n"
            f"{error}\n\n"
            "Generate a completely fresh mutation that corrects that "
            "exact problem.\n"
        )

    return prompt
