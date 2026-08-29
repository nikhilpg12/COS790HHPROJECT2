from llm_gp_hh.gp.tree import tree_to_prefix
from llm_gp_hh.llm.operators import QwenTreeOperators
from llm_gp_hh.llm.protocol import LLMClientResponse


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete_json(self, *, operation, prompt, seed):
        self.calls.append((operation, seed, prompt))
        payload = next(self.replies)
        return LLMClientResponse(data=payload, raw_text=str(payload), latency_seconds=0.01)


def test_initial_generation_accepts_only_valid_restricted_trees():
    client = FakeClient([
        {"trees": ["(+ c d)", "(if (> b e) (* c d) a)"]},
    ])
    ops = QwenTreeOperators(client=client, retry_limit=1)
    trees, records = ops.generate_initial(count=2, max_depth=4, seed=11)
    assert [tree_to_prefix(t) for t in trees] == ["(+ c d)", "(if (> b e) (* c d) a)"]
    assert len(records) == 1
    assert "Toronto terminals" in client.calls[0][2]
    assert "minimum incremental Toronto proximity penalty" in client.calls[0][2]


def test_invalid_output_is_retried_with_error_context():
    client = FakeClient([
        {"trees": ["(+ c unknown)"]},
        {"trees": ["(+ c d)"]},
    ])
    ops = QwenTreeOperators(client=client, retry_limit=2)
    trees, records = ops.generate_initial(count=1, max_depth=4, seed=5)
    assert tree_to_prefix(trees[0]) == "(+ c d)"
    assert len(records) == 2
    assert records[0].valid is False
    assert "unknown terminal" in client.calls[1][2]


def test_mutation_rejects_unchanged_parent_then_accepts_change():
    client = FakeClient([
        {"trees": ["c"]},
        {"trees": ["(+ c d)"]},
    ])
    ops = QwenTreeOperators(client=client, retry_limit=2)
    child, records = ops.mutate(
        parent="c",
        parent_metrics={"fitness": 1, "hcv": 0, "scv": 1},
        seed=9,
    )
    assert tree_to_prefix(child) == "(+ c d)"
    assert len(records) == 2
    assert "unchanged" in records[0].error


def test_crossover_returns_two_children_and_rejects_parent_copies():
    client = FakeClient([
        {"trees": ["c", "d"]},
        {"trees": ["(+ c d)", "(- c d)"]},
    ])
    ops = QwenTreeOperators(client=client, retry_limit=2)
    children, records = ops.crossover(
        parent_a="c",
        parent_a_metrics={"fitness": 2, "hcv": 0, "scv": 2},
        parent_b="d",
        parent_b_metrics={"fitness": 3, "hcv": 0, "scv": 3},
        seed=17,
    )
    assert [tree_to_prefix(x) for x in children] == ["(+ c d)", "(- c d)"]
    assert len(records) == 2
