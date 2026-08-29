import random
from pathlib import Path

from llm_gp_hh.gp.tree import parse_tree
from llm_gp_hh.toronto.constructor import construct_timetable
from llm_gp_hh.toronto.parser import load_toronto

FIXTURES = Path(__file__).parent / "fixtures"


def test_constructor_schedules_all_exams_when_feasible():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    result = construct_timetable(inst, parse_tree("c"), random.Random(7))
    assert result.hcv == 0
    assert result.unallocated == ()
    assert set(result.assignments) == set(inst.exam_ids)


def test_constructor_is_reproducible_for_same_seed():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    tree = parse_tree("(+ c d)")
    a = construct_timetable(inst, tree, random.Random(99))
    b = construct_timetable(inst, tree, random.Random(99))
    assert a == b


def test_no_feasible_period_counts_hcv_and_removes_exam(tmp_path):
    crs = tmp_path / "blocked.crs"
    stu = tmp_path / "blocked.stu"
    crs.write_text("1 1\n2 1\n", encoding="utf-8")
    stu.write_text("1 2\n", encoding="utf-8")
    inst = load_toronto(crs, stu, periods=1)
    result = construct_timetable(inst, parse_tree("c"), random.Random(1))
    assert result.hcv == 1
    assert len(result.unallocated) == 1
    assert len(result.assignments) == 1
