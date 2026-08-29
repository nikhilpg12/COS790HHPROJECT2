from __future__ import annotations

from collections.abc import Mapping


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


def initial_prompt(
    *,
    count: int,
    max_depth: int,
    error: str | None = None,
) -> str:
    """
    Build the Generation 0 prompt.

    Generation 0 candidates are checked only for GP validity.
    """
    if count < 1:
        raise ValueError(
            "count must be at least 1"
        )

    prompt = f"""You generate individuals for a Genetic Programming generation construction hyper-heuristic for Toronto examination timetabling.

You replace the conventional random GP initialisation operator.

The LLM is the sole mechanism that creates GP individuals.
The host program only parses and validates your candidates.
It does not create replacement heuristics.

You generate GP heuristic expression trees.
You do NOT generate timetables directly.

{TERMINAL_SET_TEXT}

{FUNCTION_SET_TEXT}

{TREE_FORMAT_TEXT}

Generate exactly {count} candidate GP individuals.

Requirements:

- every candidate must be a valid GP expression tree
- use only terminals from T
- use only functions from F
- respect every function's arity
- each tree must have depth at most {max_depth}
- use static and dynamic timetable attributes where useful
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
            "Correct those validation problems in the next response.\n"
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
