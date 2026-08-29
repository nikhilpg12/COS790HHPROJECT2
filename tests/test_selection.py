import random

from llm_gp_hh.gp.individual import EvaluatedIndividual
from llm_gp_hh.gp.selection import tournament_select
from llm_gp_hh.gp.tree import parse_tree


def item(identifier: str, fitness: float) -> EvaluatedIndividual:
    return EvaluatedIndividual(
        id=identifier,
        tree=parse_tree("c"),
        generation=0,
        operation="initial",
        parent_ids=(),
        hcv=0,
        scv=fitness,
        fitness=fitness,
        evaluation_seconds=0.01,
        seed=1,
    )


class SequenceRng:
    def __init__(self, values):
        self.values = iter(values)

    def randrange(self, stop):
        return next(self.values)


def test_tournament_returns_lowest_fitness_from_sample():
    pop = [item("a", 10), item("b", 2), item("c", 7)]
    chosen = tournament_select(pop, tournament_size=3, rng=SequenceRng([0, 1, 2]))
    assert chosen.id == "b"


def test_selection_with_replacement_can_pick_same_parent_repeatedly():
    pop = [item("only", 1)]
    rng = random.Random(4)
    assert tournament_select(pop, 1, rng).id == "only"
    assert tournament_select(pop, 1, rng).id == "only"
