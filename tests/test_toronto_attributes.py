from pathlib import Path

from llm_gp_hh.toronto.attributes import terminal_values
from llm_gp_hh.toronto.parser import load_toronto

FIXTURES = Path(__file__).parent / "fixtures"


def test_static_and_dynamic_toronto_terminals():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    values = terminal_values(inst, exam_id=1, assignments={2: 0}, unallocated={1, 3, 4})
    assert values["a"] == 4.0  # best current feasible distance from conflicting exam 2 is distance 3
    assert values["b"] == 1
    assert values["c"] == 2
    assert values["d"] == 2
    assert values["e"] == 3
    assert values["f"] == 1
    assert values["g"] == 4
    assert values["h"] == 4


def test_distance_terminal_is_zero_before_any_exam_is_allocated():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    values = terminal_values(inst, exam_id=1, assignments={}, unallocated=set(inst.exam_ids))
    assert values["a"] == 0.0
