from pathlib import Path
import pytest

from llm_gp_hh.toronto.fitness import proximity_cost, scalar_fitness
from llm_gp_hh.toronto.parser import load_toronto

FIXTURES = Path(__file__).parent / "fixtures"


def test_parser_builds_expected_conflicts():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    assert inst.exam_ids == (1, 2, 3, 4)
    assert inst.total_students == 4
    assert inst.conflicts[(1, 2)] == 1
    assert inst.conflicts[(1, 3)] == 1
    assert inst.conflicts[(2, 4)] == 1


def test_proximity_cost_uses_toronto_weights_and_student_normalisation():
    inst = load_toronto(FIXTURES / "toronto_tiny.crs", FIXTURES / "toronto_tiny.stu", periods=4)
    assignments = {1: 0, 2: 1, 3: 3, 4: 3}
    # (1,2):16, (1,3):4, (2,4):8 => 28 / 4 students
    assert proximity_cost(inst, assignments) == 7.0


def test_scalar_fitness_matches_paper_formula():
    assert scalar_fitness(0, 5.5) == 5.5
    assert scalar_fitness(2, 5.5) == 16.5


def test_parser_rejects_unknown_exam_id_in_student_file(tmp_path):
    crs = tmp_path / "x.crs"
    stu = tmp_path / "x.stu"
    crs.write_text("1 1\n", encoding="utf-8")
    stu.write_text("1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown exam"):
        load_toronto(crs, stu, periods=2)
