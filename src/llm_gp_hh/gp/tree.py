from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, TypeAlias

TERMINALS = frozenset("abcdefgh")
ARITHMETIC_OPS = frozenset({"+", "-", "*", "/"})
RELATIONAL_OPS = frozenset({"<", ">", "<=", ">=", "==", "!="})


@dataclass(frozen=True, slots=True)
class Terminal:
    name: str


@dataclass(frozen=True, slots=True)
class Binary:
    op: str
    left: "NumericTree"
    right: "NumericTree"


@dataclass(frozen=True, slots=True)
class Relation:
    op: str
    left: "NumericTree"
    right: "NumericTree"


@dataclass(frozen=True, slots=True)
class IfThenElse:
    condition: Relation
    then_branch: "NumericTree"
    else_branch: "NumericTree"


NumericTree: TypeAlias = Terminal | Binary | IfThenElse
Tree: TypeAlias = NumericTree | Relation

_TOKEN_RE = re.compile(r"\s*(\(|\)|<=|>=|==|!=|[+\-*/<>]|if|[A-Za-z_][A-Za-z0-9_]*)")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"invalid token near: {text[pos:pos+20]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    if not tokens:
        raise ValueError("tree expression is empty")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError("unexpected end of tree expression")
        return self.tokens[self.pos]

    def _take(self) -> str:
        token = self._peek()
        self.pos += 1
        return token

    def _expect(self, token: str) -> None:
        actual = self._take()
        if actual != token:
            raise ValueError(f"expected {token!r}, got {actual!r}")

    def parse_any(self) -> Tree:
        if self._peek() == "(":
            self._take()
            op = self._take()
            if op in ARITHMETIC_OPS:
                left = self.parse_numeric()
                right = self.parse_numeric()
                self._expect(")")
                return Binary(op, left, right)
            if op in RELATIONAL_OPS:
                left = self.parse_numeric()
                right = self.parse_numeric()
                self._expect(")")
                return Relation(op, left, right)
            if op == "if":
                condition = self.parse_relation()
                then_branch = self.parse_numeric()
                else_branch = self.parse_numeric()
                self._expect(")")
                return IfThenElse(condition, then_branch, else_branch)
            raise ValueError(f"unknown function {op!r}")
        terminal = self._take()
        if terminal not in TERMINALS:
            raise ValueError(f"unknown terminal {terminal!r}")
        return Terminal(terminal)

    def parse_numeric(self) -> NumericTree:
        node = self.parse_any()
        if isinstance(node, Relation):
            raise ValueError("relational expressions are only valid as if conditions")
        return node

    def parse_relation(self) -> Relation:
        node = self.parse_any()
        if not isinstance(node, Relation):
            raise ValueError("if condition must be a relational expression")
        return node


def parse_tree(text: str) -> Tree:
    parser = _Parser(_tokenize(text))
    tree = parser.parse_any()
    if parser.pos != len(parser.tokens):
        raise ValueError("unexpected trailing tokens")
    return tree


def tree_to_prefix(tree: Tree) -> str:
    if isinstance(tree, Terminal):
        return tree.name
    if isinstance(tree, Binary):
        return f"({tree.op} {tree_to_prefix(tree.left)} {tree_to_prefix(tree.right)})"
    if isinstance(tree, Relation):
        return f"({tree.op} {tree_to_prefix(tree.left)} {tree_to_prefix(tree.right)})"
    if isinstance(tree, IfThenElse):
        return (
            f"(if {tree_to_prefix(tree.condition)} "
            f"{tree_to_prefix(tree.then_branch)} {tree_to_prefix(tree.else_branch)})"
        )
    raise TypeError(f"unsupported tree node: {type(tree)!r}")


def tree_depth(tree: Tree) -> int:
    if isinstance(tree, Terminal):
        return 1
    if isinstance(tree, (Binary, Relation)):
        return 1 + max(tree_depth(tree.left), tree_depth(tree.right))
    if isinstance(tree, IfThenElse):
        return 1 + max(
            tree_depth(tree.condition),
            tree_depth(tree.then_branch),
            tree_depth(tree.else_branch),
        )
    raise TypeError(f"unsupported tree node: {type(tree)!r}")


def node_count(tree: Tree) -> int:
    if isinstance(tree, Terminal):
        return 1
    if isinstance(tree, (Binary, Relation)):
        return 1 + node_count(tree.left) + node_count(tree.right)
    if isinstance(tree, IfThenElse):
        return 1 + node_count(tree.condition) + node_count(tree.then_branch) + node_count(tree.else_branch)
    raise TypeError(f"unsupported tree node: {type(tree)!r}")


def validate_tree(tree: Tree, max_depth: int | None = None) -> None:
    if isinstance(tree, Relation):
        raise ValueError("heuristic must have a numeric root, not a relational root")
    if max_depth is not None and tree_depth(tree) > max_depth:
        raise ValueError(f"tree exceeds maximum depth {max_depth}")
    _validate_numeric(tree)


def _validate_numeric(tree: NumericTree) -> None:
    if isinstance(tree, Terminal):
        if tree.name not in TERMINALS:
            raise ValueError(f"unknown terminal {tree.name!r}")
        return
    if isinstance(tree, Binary):
        if tree.op not in ARITHMETIC_OPS:
            raise ValueError(f"unknown arithmetic function {tree.op!r}")
        _validate_numeric(tree.left)
        _validate_numeric(tree.right)
        return
    if isinstance(tree, IfThenElse):
        if tree.condition.op not in RELATIONAL_OPS:
            raise ValueError(f"unknown relational function {tree.condition.op!r}")
        _validate_numeric(tree.condition.left)
        _validate_numeric(tree.condition.right)
        _validate_numeric(tree.then_branch)
        _validate_numeric(tree.else_branch)
        return
    raise ValueError("relational expressions are only valid as if conditions")


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def evaluate_tree(tree: Tree, terminals: Mapping[str, float]) -> float:
    validate_tree(tree)
    return _eval_numeric(tree, terminals)


def _terminal_value(name: str, terminals: Mapping[str, float]) -> float:
    if name not in terminals:
        raise KeyError(f"missing terminal value {name!r}")
    return float(terminals[name])


def _eval_numeric(tree: NumericTree, terminals: Mapping[str, float]) -> float:
    if isinstance(tree, Terminal):
        return _terminal_value(tree.name, terminals)
    if isinstance(tree, Binary):
        left = _eval_numeric(tree.left, terminals)
        right = _eval_numeric(tree.right, terminals)
        try:
            if tree.op == "+":
                return _finite(left + right)
            if tree.op == "-":
                return _finite(left - right)
            if tree.op == "*":
                return _finite(left * right)
            if tree.op == "/":
                return 1.0 if right == 0 else _finite(left / right)
        except OverflowError:
            return 0.0
        raise ValueError(f"unknown arithmetic function {tree.op!r}")
    if isinstance(tree, IfThenElse):
        branch = tree.then_branch if _eval_relation(tree.condition, terminals) else tree.else_branch
        return _eval_numeric(branch, terminals)
    raise TypeError("relational expression cannot be evaluated as a numeric heuristic")


def _eval_relation(tree: Relation, terminals: Mapping[str, float]) -> bool:
    left = _eval_numeric(tree.left, terminals)
    right = _eval_numeric(tree.right, terminals)
    if tree.op == "<":
        return left < right
    if tree.op == ">":
        return left > right
    if tree.op == "<=":
        return left <= right
    if tree.op == ">=":
        return left >= right
    if tree.op == "==":
        return left == right
    if tree.op == "!=":
        return left != right
    raise ValueError(f"unknown relational function {tree.op!r}")
