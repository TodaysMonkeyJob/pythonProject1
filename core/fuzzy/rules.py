"""
Rule base (F15 / TOD-26): ``IF <antecedent> THEN <variable> IS <term>`` with optional weight.

Antecedent grammar (keywords case-insensitive; AND binds tighter than OR)::

    expr   := and ("OR" and)*
    and    := unary ("AND" unary)*
    unary  := "NOT" unary | "(" expr ")" | clause
    clause := variable "IS" ["NOT"] term

In YAML a rule is a string or ``{id, rule, weight}``; ``"… WITH 0.5"`` at the end is a shorthand for the weight.
"""
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

Node = Union[Tuple[str, str, str], Tuple[str, Any]]

TOKEN = re.compile(r"\(|\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S")
KEYWORDS = {"IF", "THEN", "IS", "AND", "OR", "NOT", "WITH"}

T_NORMS: Dict[str, Callable[[float, float], float]] = {"min": min, "prod": lambda a, b: a * b}
S_NORMS: Dict[str, Callable[[float, float], float]] = {"max": max, "probor": lambda a, b: a + b - a * b}


class RuleError(ValueError):
    pass


@dataclass
class Rule:
    id: str
    antecedent: Node
    output: str
    term: str
    weight: float = 1.0
    source: str = "base"          # base | location override (explanations: why the bound is what it is)

    @property
    def text(self) -> str:
        return f"IF {render(self.antecedent)} THEN {self.output} IS {self.term}"

    def clauses(self) -> List[Tuple[str, str]]:
        """:return: Every (variable, term) the antecedent uses"""
        out: List[Tuple[str, str]] = []
        _walk(self.antecedent, out)
        return out

    def literals(self) -> List[Tuple[str, str, bool]]:
        """:return: (variable, term, negated) of every clause — explanations"""
        out: List[Tuple[str, str, bool]] = []
        _literals(self.antecedent, False, out)
        return out

    def variables(self) -> Set[str]:
        return {var for var, _ in self.clauses()}

    def strength(self, memberships: Dict[str, Dict[str, float]], t_norm: str = "min", s_norm: str = "max") -> float:
        """:param memberships: {variable: {term: μ}} of the current inputs. :return: firing strength × weight"""
        return evaluate(self.antecedent, memberships, T_NORMS[t_norm], S_NORMS[s_norm]) * self.weight


def _walk(node: Node, out: List[Tuple[str, str]]) -> None:
    kind = node[0]
    if kind == "is":
        out.append((node[1], node[2]))
    elif kind == "not":
        _walk(node[1], out)
    else:
        for child in node[1]:
            _walk(child, out)


def _literals(node: Node, negated: bool, out: List[Tuple[str, str, bool]]) -> None:
    kind = node[0]
    if kind == "is":
        out.append((node[1], node[2], negated))
    elif kind == "not":
        _literals(node[1], not negated, out)
    else:
        for child in node[1]:
            _literals(child, negated, out)


def evaluate(node: Node, memberships: Dict[str, Dict[str, float]], t_norm, s_norm) -> float:
    kind = node[0]
    if kind == "is":
        return memberships[node[1]][node[2]]
    if kind == "not":
        return 1.0 - evaluate(node[1], memberships, t_norm, s_norm)
    values = [evaluate(child, memberships, t_norm, s_norm) for child in node[1]]
    result = values[0]
    for value in values[1:]:
        result = (t_norm if kind == "and" else s_norm)(result, value)
    return result


def render(node: Node, parent: Optional[str] = None) -> str:
    kind = node[0]
    if kind == "is":
        return f"{node[1]} IS {node[2]}"
    if kind == "not":
        inner = node[1]
        if inner[0] == "is":
            return f"{inner[1]} IS NOT {inner[2]}"
        return f"NOT ({render(inner)})"
    text = f" {kind.upper()} ".join(render(child, kind) for child in node[1])
    return f"({text})" if parent == "and" and kind == "or" else text


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = TOKEN.findall(text)
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def keyword(self, word: str) -> bool:
        token = self.peek()
        if token is not None and token.upper() == word:
            self.i += 1
            return True
        return False

    def expect(self, word: str) -> None:
        if not self.keyword(word):
            raise RuleError(f"expected {word} at token {self.i + 1} ({self.peek()!r}) in {self.text!r}")

    def name(self, what: str) -> str:
        token = self.peek()
        if token is None or not re.match(r"^[A-Za-z_]\w*$", token) or token.upper() in KEYWORDS:
            raise RuleError(f"expected a {what} name at token {self.i + 1} ({token!r}) in {self.text!r}")
        self.i += 1
        return token

    def expr(self) -> Node:
        parts = [self.conjunction()]
        while self.keyword("OR"):
            parts.append(self.conjunction())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def conjunction(self) -> Node:
        parts = [self.unary()]
        while self.keyword("AND"):
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def unary(self) -> Node:
        if self.keyword("NOT"):
            return ("not", self.unary())
        if self.peek() == "(":
            self.i += 1
            node = self.expr()
            if self.peek() != ")":
                raise RuleError(f"missing ')' in {self.text!r}")
            self.i += 1
            return node
        variable = self.name("variable")
        self.expect("IS")
        negate = self.keyword("NOT")
        node: Node = ("is", variable, self.name("term"))
        return ("not", node) if negate else node


def parse_rule(spec: Any, index: int, source: str = "base") -> Rule:
    """
    :param spec: ``"IF … THEN out IS term [WITH w]"`` or ``{id, rule, weight}``
    :param index: Position (default id ``R<index>``)
    """
    data = spec if isinstance(spec, dict) else {"rule": spec}
    text = str(data.get("rule", "")).strip()
    parser = _Parser(text)
    parser.expect("IF")
    antecedent = parser.expr()
    parser.expect("THEN")
    output = parser.name("output variable")
    parser.expect("IS")
    term = parser.name("output term")
    weight = float(data.get("weight", 1.0))
    if parser.keyword("WITH"):
        token = parser.peek()
        try:
            weight = float(token)
        except (TypeError, ValueError) as e:
            raise RuleError(f"WITH needs a number in {text!r}") from e
        parser.i += 1
    if parser.peek() is not None:
        raise RuleError(f"unexpected {parser.peek()!r} at the end of {text!r}")
    if not 0.0 <= weight <= 1.0:
        raise RuleError(f"weight must be within 0..1, got {weight} in {text!r}")
    return Rule(str(data.get("id") or f"R{index}"), antecedent, output, term, weight, source)
