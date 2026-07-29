from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

# Tokenizer: a NUMBER, an IDENT (variable/function name, including Maxima's `%`-prefixed
# constants like %pi/%e/%i), or a single-character operator/punctuation. Leading
# whitespace before each token is consumed inline via `\s*` so the scanner doesn't need
# a separate whitespace-skipping pass.
_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<NUMBER>\d+\.\d+|\d+)
      | (?P<IDENT>[A-Za-z_%][A-Za-z0-9_%]*)
      | (?P<OP>[+\-*/^,()])
    )""",
    re.VERBOSE,
)


@dataclass
class ExprNode:
    """One node of a parsed CAS expression tree: `label` is the operator/function/atom
    name, `children` are its operands in left-to-right order (empty for a leaf)."""

    label: str
    children: list["ExprNode"] = field(default_factory=list)


class _ParseError(Exception):
    pass


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            if text[pos:].strip() == "":
                break
            raise _ParseError(f"Unexpected character {text[pos]!r} at position {pos}")
        kind = match.lastgroup
        value = match.group(kind)
        tokens.append((kind, value))
        pos = match.end()
    return tokens


class _Parser:
    """Recursive-descent parser for a Maxima-style arithmetic expression.

    Grammar (highest precedence last):
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := '-' factor | power
        power   := atom ('^' factor)?              # right-associative
        atom    := NUMBER | IDENT ['(' arglist ')'] | '(' expr ')'
        arglist := expr (',' expr)*

    `+` and `*` chains flatten into one n-ary node when every operator in the run is the
    same (commutative/associative), matching the expression-tree shape STACK/Maxima
    itself produces (e.g. `2*x*sqrt(x)` -> one `*` node with 3 children, not nested
    binary nodes) — this is exactly the shape shown in Takada et al. (EDM 2025) Figure 2.
    A mixed +/- or */÷ run instead folds left-to-right into nested binary nodes using the
    actual operator at each step, since subtraction/division aren't associative.
    A unary minus applied directly to a bare numeric literal (not itself raised to a
    power) folds into the literal as one atom (e.g. `-1`), per that same paper's stated
    convention ("we regard '-1' an atom"); a unary minus on anything else becomes a
    unary `-` node wrapping its operand.
    """

    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self, offset: int = 0) -> tuple[str, str] | None:
        index = self._pos + offset
        return self._tokens[index] if index < len(self._tokens) else None

    def _peek_is(self, value: str, offset: int = 0) -> bool:
        token = self._peek(offset)
        return token is not None and token[1] == value

    def _advance(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise _ParseError("Unexpected end of expression")
        self._pos += 1
        return token

    def _expect(self, value: str) -> None:
        if not self._peek_is(value):
            raise _ParseError(f"Expected {value!r} at position {self._pos}")
        self._advance()

    def parse_expr(self) -> ExprNode:
        operands = [self.parse_term()]
        ops: list[str] = []
        while self._peek_is("+") or self._peek_is("-"):
            ops.append(self._advance()[1])
            operands.append(self.parse_term())
        if not ops:
            return operands[0]
        if all(op == "+" for op in ops):
            return ExprNode("+", operands)
        acc = operands[0]
        for op, rhs in zip(ops, operands[1:]):
            acc = ExprNode(op, [acc, rhs])
        return acc

    def parse_term(self) -> ExprNode:
        operands = [self.parse_factor()]
        ops: list[str] = []
        while self._peek_is("*") or self._peek_is("/"):
            ops.append(self._advance()[1])
            operands.append(self.parse_factor())
        if not ops:
            return operands[0]
        if all(op == "*" for op in ops):
            return ExprNode("*", operands)
        acc = operands[0]
        for op, rhs in zip(ops, operands[1:]):
            acc = ExprNode(op, [acc, rhs])
        return acc

    def parse_factor(self) -> ExprNode:
        if self._peek_is("-"):
            self._advance()
            next_token = self._peek()
            if next_token is not None and next_token[0] == "NUMBER" and not self._peek_is("^", 1):
                value = self._advance()[1]
                return ExprNode(f"-{value}")
            return ExprNode("-", [self.parse_factor()])
        return self.parse_power()

    def parse_power(self) -> ExprNode:
        base = self.parse_atom()
        if self._peek_is("^"):
            self._advance()
            exponent = self.parse_factor()
            return ExprNode("^", [base, exponent])
        return base

    def parse_atom(self) -> ExprNode:
        token = self._peek()
        if token is None:
            raise _ParseError("Unexpected end of expression")
        kind, value = token
        if kind == "NUMBER":
            self._advance()
            return ExprNode(value)
        if kind == "IDENT":
            self._advance()
            if self._peek_is("("):
                self._advance()
                args = self.parse_arglist()
                self._expect(")")
                return ExprNode(value, args)
            return ExprNode(value)
        if kind == "OP" and value == "(":
            self._advance()
            node = self.parse_expr()
            self._expect(")")
            return node
        raise _ParseError(f"Unexpected token {value!r}")

    def parse_arglist(self) -> list[ExprNode]:
        if self._peek_is(")"):
            return []
        args = [self.parse_expr()]
        while self._peek_is(","):
            self._advance()
            args.append(self.parse_expr())
        return args


@functools.lru_cache(maxsize=4096)
def parse_expression(text: str) -> ExprNode | None:
    """Parse a Maxima/STACK CAS expression string into an `ExprNode` tree, or `None` if
    it can't be confidently parsed (empty input, or syntax outside the supported
    arithmetic/function-call grammar, e.g. `matrix(...)`) — callers should treat `None`
    as "no tree edit distance available for this response" rather than raising."""
    if not text or not text.strip():
        return None
    try:
        tokens = _tokenize(text)
        if not tokens:
            return None
        parser = _Parser(tokens)
        node = parser.parse_expr()
        if parser._pos != len(parser._tokens):
            return None
        return node
    except _ParseError:
        return None
