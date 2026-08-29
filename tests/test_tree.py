import pytest

from llm_gp_hh.gp.tree import (
    evaluate_tree,
    node_count,
    parse_tree,
    tree_depth,
    tree_to_prefix,
    validate_tree,
)


def test_parse_round_trip_arithmetic_tree():
    tree = parse_tree("(+ (* c d) f)")
    assert tree_to_prefix(tree) == "(+ (* c d) f)"
    assert tree_depth(tree) == 3
    assert node_count(tree) == 5


def test_protected_division_returns_one_for_zero_denominator():
    tree = parse_tree("(/ c e)")
    assert evaluate_tree(tree, {"c": 7, "e": 0}) == 1.0


def test_conditional_rule_evaluates_selected_branch():
    tree = parse_tree("(if (> b e) (* c d) a)")
    values = {"a": 2, "b": 5, "c": 3, "d": 4, "e": 1}
    assert evaluate_tree(tree, values) == 12


def test_unknown_terminal_is_rejected():
    with pytest.raises(ValueError, match="unknown terminal"):
        parse_tree("(+ c z)")


def test_relational_node_cannot_be_root_numeric_heuristic():
    tree = parse_tree("(> b e)")
    with pytest.raises(ValueError, match="numeric root"):
        validate_tree(tree)


def test_relation_is_only_valid_as_if_condition():
    with pytest.raises(ValueError, match="relational"):
        parse_tree("(+ (> b e) c)")


def test_max_depth_validation():
    tree = parse_tree("(+ (* c d) f)")
    with pytest.raises(ValueError, match="maximum depth"):
        validate_tree(tree, max_depth=2)
