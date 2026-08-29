from pathlib import Path

from llm_gp_hh.config import RunConfig
from llm_gp_hh.gp.evolution import EvolutionEngine
from llm_gp_hh.gp.tree import parse_tree
from llm_gp_hh.toronto.parser import load_toronto

FIXTURES = Path(__file__).parent / "fixtures"


class MockOperators:
    def __init__(self):
        self.crossover_calls = 0
        self.mutation_calls = 0

    def generate_initial(self, *, count, max_depth, seed):
        pool = [parse_tree(x) for x in ["c", "d", "(+ c d)", "e"]]
        return pool[:count], []

    def crossover(self, *, parent_a, parent_a_metrics, parent_b, parent_b_metrics, seed):
        self.crossover_calls += 1
        return (parse_tree("(+ c d)"), parse_tree("(+ b e)")), []

    def mutate(self, *, parent, parent_metrics, seed):
        self.mutation_calls += 1
        return parse_tree("(+ c e)"), []


def test_engine_runs_two_generations_and_preserves_population_size():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    cfg = RunConfig(population_size=4, generations=2, tournament_size=2, seed=123)
    result = EvolutionEngine().run(inst, MockOperators(), cfg, seed=123)
    assert len(result.generations) == 2
    assert all(len(g.population) == 4 for g in result.generations)
    assert result.best.fitness == min(i.fitness for g in result.generations for i in g.population)
    assert all(i.seed >= 0 for g in result.generations for i in g.population)


def test_all_mutation_rate_uses_only_mutation():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    cfg = RunConfig(
        population_size=4,
        generations=2,
        tournament_size=2,
        crossover_rate=0.0,
        mutation_rate=1.0,
        seed=7,
    )
    ops = MockOperators()
    result = EvolutionEngine().run(inst, ops, cfg, seed=7)
    assert ops.crossover_calls == 0
    assert ops.mutation_calls == 4
    assert result.generations[1].operator_counts == {"crossover": 0, "mutation": 4}


def test_repeated_run_with_same_seed_and_mock_operators_is_reproducible():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    cfg = RunConfig(population_size=4, generations=2, tournament_size=2, seed=42)
    first = EvolutionEngine().run(inst, MockOperators(), cfg, seed=42)
    second = EvolutionEngine().run(inst, MockOperators(), cfg, seed=42)
    assert [i.tree for g in first.generations for i in g.population] == [i.tree for g in second.generations for i in g.population]
    assert [i.fitness for g in first.generations for i in g.population] == [i.fitness for g in second.generations for i in g.population]


def test_cli_accepts_required_toronto_inputs():
    from llm_gp_hh.experiments.run import build_parser

    args = build_parser().parse_args([
        "--crs", "x.crs",
        "--stu", "x.stu",
        "--periods", "18",
        "--profile", "dev",
        "--seed", "123",
    ])
    assert args.periods == 18
    assert args.seed == 123
    assert args.profile == "dev"


def test_candidate_record_contains_tree_shape_metadata():
    from llm_gp_hh.experiments.run import candidate_record
    from llm_gp_hh.gp.individual import EvaluatedIndividual

    item = EvaluatedIndividual(
        id="g0-i0",
        tree=parse_tree("(+ c d)"),
        generation=0,
        operation="initial",
        parent_ids=(),
        seed=123,
        hcv=0,
        scv=4.5,
        fitness=4.5,
        evaluation_seconds=0.01,
    )
    record = candidate_record(item)
    assert record["tree"] == "(+ c d)"
    assert record["tree_depth"] == 2
    assert record["node_count"] == 3


def test_evolution_annotates_llm_calls_with_generation_and_parents():
    from llm_gp_hh.llm.protocol import LLMCallRecord

    class RecordingOperators(MockOperators):
        def generate_initial(self, *, count, max_depth, seed):
            trees, _ = super().generate_initial(count=count, max_depth=max_depth, seed=seed)
            return trees, [LLMCallRecord(
                operation="initial", seed=seed, prompt="p", raw_response="r",
                latency_seconds=0.01, valid=True
            )]

        def mutate(self, *, parent, parent_metrics, seed):
            tree, _ = super().mutate(parent=parent, parent_metrics=parent_metrics, seed=seed)
            return tree, [LLMCallRecord(
                operation="mutation", seed=seed, prompt="p", raw_response="r",
                latency_seconds=0.01, valid=True
            )]

    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    cfg = RunConfig(
        population_size=4, generations=2, tournament_size=2,
        crossover_rate=0.0, mutation_rate=1.0, seed=9,
    )
    result = EvolutionEngine().run(inst, RecordingOperators(), cfg, seed=9)
    assert result.llm_calls[0].generation == 0
    later = [call for call in result.llm_calls if call.operation == "mutation"]
    assert later and all(call.generation == 1 for call in later)
    assert all(len(call.parent_ids) == 1 for call in later)
