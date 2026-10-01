"""
Natural-language boolean formula parser and verifier.

Expected format (inside <cnf>...</cnf> tags):

    (x1 OR NOT x2 OR x3) AND (NOT x1 OR x2) AND (x2 OR x3)

Grammar:
    nat_formula ::= nat_and
    nat_and     ::= nat_or (AND nat_or)*
    nat_or      ::= nat_not (OR nat_not)*
    nat_not     ::= NOT nat_not | nat_atom
    nat_atom    ::= variable | "(" nat_and ")"
    variable    ::= [A-Za-z_][A-Za-z0-9_]*

Operators are AND, OR, NOT, matched case-insensitively (``and``/``And``/``AND``
all work).  Symbolic (&, |, ~), Unicode (∧, ∨, ¬), and other boolean operators
(XOR, NAND, NOR, XNOR) are not supported and are flagged as
``invalid_operator`` errors regardless of case.

Strict CNF clauses are extracted directly from the parsed AST. General boolean
formulas are first reduced to strict CNF using De Morgan's laws (``_to_nnf``)
followed by OR-over-AND distribution (``_cnf_distribute``). No auxiliary
variables are introduced. Literals are then resolved either through the
target-supplied ordered mapping or, for legacy standalone calls, by first
appearance, and the resulting integer clauses are fed into the standard
DIMACS parser.
"""

from __future__ import annotations

import itertools
import re
from typing import List, Optional, Sequence, Tuple

from pysat.formula import (
    Atom as PySATAtom,
    And as PySATAnd,
    Or as PySATOr,
    Neg as PySATNeg,
)

from fonesy.evaluation.dimacs import parse_dimacs
from fonesy.evaluation.formula import CNFFormula, normalize_variable_names
from fonesy.statistics.statistics import CorrectionStatus

# ---------------------------------------------------------------------------
# Error-type classification
# ---------------------------------------------------------------------------

_NAT_ERROR_PATTERNS: List[Tuple[str, str]] = [
    ("no_cnf_tags",       "Could not find <cnf>"),
    ("empty_formula",     "Empty formula"),
    ("parse_error",       "Parse error"),
    ("invalid_operator",  "Invalid operator"),
    ("empty_clause",      "Empty clause"),
    ("unknown_variable",  "Unknown variable"),
    ("formula_too_complex", "Formula too complex"),
]

NAT_ERROR_TYPE_CODES: List[str] = [code for code, _ in _NAT_ERROR_PATTERNS]


def classify_nat_error_types(error_message: str) -> List[str]:
    """Return nat error-type codes present in *error_message*."""
    return [code for code, pattern in _NAT_ERROR_PATTERNS if pattern in error_message]


_NAT_CODE_EXPLANATIONS = {
    "no_cnf_tags":      "missing <cnf>…</cnf> wrapper",
    "empty_formula":    "nothing between the <cnf> tags",
    "parse_error":      "formula could not be parsed (bad syntax)",
    "invalid_operator": "disallowed operator (must use AND, OR, NOT — any letter case)",
    "empty_clause":     "empty clause () in formula",
    "unknown_variable": "a formula symbol is not in the expected variable mapping",
    "formula_too_complex": "general formula too large to distribute into strict CNF",
}

_NAT_CODE_GUIDANCE = {
    "no_cnf_tags": (
        "Wrap your entire formula in <cnf>...</cnf> tags — "
        "the very first token must be <cnf> and the last must be </cnf>."
    ),
    "empty_formula": (
        "Write at least one clause between the <cnf>...</cnf> tags, "
        "e.g. <cnf> (x1 OR NOT x2) </cnf>."
    ),
    "parse_error": (
        "Use variables joined by OR/AND/NOT (any letter case is fine).  "
        "Example: (x1 OR NOT x2) AND (NOT x1 OR x2 OR x3)."
    ),
    "invalid_operator": (
        "Use ONLY the operators AND, OR, NOT (any letter case is fine, e.g. 'and' works).  "
        "Symbolic (&/|/~), Unicode (∧/∨/¬), and XOR/NAND/NOR/XNOR are not allowed."
    ),
    "empty_clause": (
        "Remove empty parentheses ().  Every clause must contain at least one literal."
    ),
    "unknown_variable": (
        "Use only variable names from the ordered mapping supplied in the prompt."
    ),
    "formula_too_complex": (
        "Rewrite the formula as an explicit AND-of-OR (strict CNF) directly, "
        "e.g. (x1 OR NOT x2) AND (x2 OR x3) — a wide OR of large AND groups "
        "cannot be distributed into strict CNF without exploding in size."
    ),
}

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class _NatSyntaxError(Exception):
    """Raised during tokenization or parsing; carries the nat error code."""
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

_TK_AND   = "AND"
_TK_OR    = "OR"
_TK_NOT   = "NOT"
_TK_LPAREN = "LPAREN"
_TK_RPAREN = "RPAREN"
_TK_IDENT  = "IDENT"
_TK_EOF    = "EOF"

# Patterns that indicate disallowed operators (checked before IDENT so they're
# caught early and produce a clean error message). AND/OR/NOT are handled
# case-insensitively by _TOKEN_RE directly, so they are not flagged here;
# every operator below is unsupported regardless of case.
_INVALID_OPERATOR_PATTERNS = [
    (re.compile(r'[&|~]'), "symbolic operator"),
    (re.compile(r'[∧∨¬⊕]'), "Unicode operator"),
    (re.compile(r'(?<![A-Za-z0-9_])XOR(?![A-Za-z0-9_])', re.IGNORECASE), "XOR operator"),
    (re.compile(r'(?<![A-Za-z0-9_])NAND(?![A-Za-z0-9_])', re.IGNORECASE), "NAND operator"),
    (re.compile(r'(?<![A-Za-z0-9_])NOR(?![A-Za-z0-9_])', re.IGNORECASE), "NOR operator"),
    (re.compile(r'(?<![A-Za-z0-9_])XNOR(?![A-Za-z0-9_])', re.IGNORECASE), "XNOR operator"),
]

_TOKEN_RE = re.compile(
    r"(?P<AND>\b(?i:AND)\b)"
    r"|(?P<OR>\b(?i:OR)\b)"
    r"|(?P<NOT>\b(?i:NOT)\b)"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<WS>\s+)"
)


def _check_invalid_operators(text: str) -> None:
    """Raise _NatSyntaxError if *text* contains any disallowed operator."""
    for pattern, label in _INVALID_OPERATOR_PATTERNS:
        m = pattern.search(text)
        if m:
            raise _NatSyntaxError(
                "invalid_operator",
                f"Invalid operator: '{m.group()}' is a {label}; use AND, OR, NOT (uppercase ASCII only)",
            )


def _tokenize(text: str) -> List[Tuple[str, str]]:
    """Return list of (token_type, value) pairs; raises _NatSyntaxError on bad input."""
    _check_invalid_operators(text)
    tokens = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise _NatSyntaxError("parse_error", f"Parse error: unexpected character '{text[pos]}' at position {pos}")
        kind = m.lastgroup
        if kind != "WS":
            tokens.append((kind, m.group()))
        pos = m.end()
    tokens.append((_TK_EOF, ""))
    return tokens


# ---------------------------------------------------------------------------
# AST node types (our own, lightweight)
# ---------------------------------------------------------------------------

class _Atom:
    __slots__ = ("name",)
    def __init__(self, name: str): self.name = name

class _Not:
    __slots__ = ("child",)
    def __init__(self, child): self.child = child

class _And:
    __slots__ = ("children",)
    def __init__(self, children): self.children = children

class _Or:
    __slots__ = ("children",)
    def __init__(self, children): self.children = children


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens):
        self._tokens = tokens
        self._pos = 0

    def _peek(self):
        return self._tokens[self._pos][0]

    def _consume(self, expected=None):
        kind, val = self._tokens[self._pos]
        if expected and kind != expected:
            raise _NatSyntaxError(
                "parse_error",
                f"Parse error: expected {expected} but got '{val}' ({kind})",
            )
        self._pos += 1
        return kind, val

    def parse(self):
        node = self._parse_and()
        self._consume(_TK_EOF)
        return node

    def _parse_and(self):
        children = [self._parse_or()]
        while self._peek() == _TK_AND:
            self._consume(_TK_AND)
            children.append(self._parse_or())
        return children[0] if len(children) == 1 else _And(children)

    def _parse_or(self):
        children = [self._parse_not()]
        while self._peek() == _TK_OR:
            self._consume(_TK_OR)
            children.append(self._parse_not())
        return children[0] if len(children) == 1 else _Or(children)

    def _parse_not(self):
        if self._peek() == _TK_NOT:
            self._consume(_TK_NOT)
            return _Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self):
        if self._peek() == _TK_LPAREN:
            self._consume(_TK_LPAREN)
            if self._peek() == _TK_RPAREN:
                raise _NatSyntaxError("empty_clause", "Empty clause: () is not a valid clause")
            node = self._parse_and()
            self._consume(_TK_RPAREN)
            return node
        if self._peek() == _TK_IDENT:
            _, val = self._consume(_TK_IDENT)
            return _Atom(val)
        kind, val = self._tokens[self._pos]
        raise _NatSyntaxError(
            "parse_error",
            f"Parse error: unexpected token '{val}' ({kind}); expected variable or '('",
        )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _first_appearance_order(node, seen=None, result=None):
    """Return variable names in first-appearance (left-to-right) order."""
    if seen is None:
        seen = set()
        result = []
    if isinstance(node, _Atom):
        if node.name not in seen:
            seen.add(node.name)
            result.append(node.name)
    elif isinstance(node, _Not):
        _first_appearance_order(node.child, seen, result)
    elif isinstance(node, (_And, _Or)):
        for c in node.children:
            _first_appearance_order(c, seen, result)
    return result


def _is_literal(node):
    return isinstance(node, _Atom) or (isinstance(node, _Not) and isinstance(node.child, _Atom))


def _is_strict_cnf(node):
    """True if node is And(Or(literal,...), ...) or a single Or(literal,...) or literal."""
    if isinstance(node, _And):
        return all(_is_strict_cnf(c) for c in node.children)
    if isinstance(node, _Or):
        return all(_is_literal(c) for c in node.children)
    return _is_literal(node)


def _ast_to_pysat(node, atom_map):
    """Convert custom AST node to a PySAT formula object (Atom/And/Or/Neg)."""
    if isinstance(node, _Atom):
        if node.name not in atom_map:
            atom_map[node.name] = PySATAtom(node.name)
        return atom_map[node.name]
    if isinstance(node, _Not):
        return PySATNeg(_ast_to_pysat(node.child, atom_map))
    if isinstance(node, _Or):
        return PySATOr(*[_ast_to_pysat(c, atom_map) for c in node.children])
    if isinstance(node, _And):
        return PySATAnd(*[_ast_to_pysat(c, atom_map) for c in node.children])
    raise ValueError(f"Unexpected AST node type: {type(node)}")


def _extract_strict_clauses(pysat_formula):
    """Extract integer clauses from a strict-CNF PySAT formula without aux variables.

    Calls ``clausify()`` directly on each top-level subformula (Or / literal)
    so no Tseitin auxiliary variables are introduced.
    """
    if isinstance(pysat_formula, PySATAnd):
        clauses = []
        for sf in pysat_formula.subformulas:
            sf.clausify()
            clauses.extend(sf.clauses)
        return clauses
    pysat_formula.clausify()
    return list(pysat_formula.clauses)


def _to_nnf(node):
    """Convert AST to Negation Normal Form (push negations to atoms via De Morgan)."""
    if isinstance(node, _Atom):
        return node
    if isinstance(node, _Not):
        child = node.child
        if isinstance(child, _Atom):
            return node  # ~x is already a literal
        if isinstance(child, _Not):
            return _to_nnf(child.child)  # ~~x → x
        if isinstance(child, _And):
            # ~(A ∧ B) → (~A ∨ ~B)
            return _to_nnf(_Or([_Not(c) for c in child.children]))
        if isinstance(child, _Or):
            # ~(A ∨ B) → (~A ∧ ~B)
            return _to_nnf(_And([_Not(c) for c in child.children]))
    if isinstance(node, _And):
        return _And([_to_nnf(c) for c in node.children])
    if isinstance(node, _Or):
        return _Or([_to_nnf(c) for c in node.children])
    return node


# Maximum number of clauses produced by distributing OR over AND; larger
# conversions are reported as `formula_too_complex` instead of running for
# hours.
_MAX_CNF_DISTRIBUTE_CLAUSES = 50_000


def _cnf_distribute(node, _budget=None):
    """Convert an NNF AST to a list of clauses (each clause = list of literals).

    Each literal is an ``_Atom`` or ``_Not(_Atom)`` node.  Uses distributivity
    (OR over AND) to produce strict CNF; tautological clauses are discarded.
    The result may be exponentially larger than the original for deeply
    nested formulas — safe for the small formulas LLMs typically produce,
    but raises :class:`_NatSyntaxError` (code ``formula_too_complex``) once
    ``_MAX_CNF_DISTRIBUTE_CLAUSES`` is exceeded rather than continuing an
    effectively unbounded computation.
    """
    if _budget is None:
        _budget = [_MAX_CNF_DISTRIBUTE_CLAUSES]
    if _is_literal(node):
        return [[node]]
    if isinstance(node, _And):
        result = []
        for child in node.children:
            result.extend(_cnf_distribute(child, _budget))
        return result
    if isinstance(node, _Or):
        child_cnfs = [_cnf_distribute(c, _budget) for c in node.children]
        result = []
        for combo in itertools.product(*child_cnfs):
            merged = [lit for clause in combo for lit in clause]
            # Discard tautological clauses (x and ~x in the same clause)
            pos = {lit.name for lit in merged if isinstance(lit, _Atom)}
            neg = {lit.child.name for lit in merged if not isinstance(lit, _Atom)}
            if pos & neg:
                continue
            result.append(merged)
            _budget[0] -= 1
            if _budget[0] <= 0:
                raise _NatSyntaxError(
                    "formula_too_complex",
                    "General boolean formula too large to distribute into "
                    f"strict CNF without auxiliary variables (exceeded "
                    f"{_MAX_CNF_DISTRIBUTE_CLAUSES} clauses during OR-over-AND "
                    "distribution)",
                )
        return result
    return [[node]]


def _strict_clause_lists(node):
    """Return literal-node clauses from an AST known to be strict CNF."""
    if isinstance(node, _And):
        clauses = []
        for child in node.children:
            clauses.extend(_strict_clause_lists(child))
        return clauses
    if isinstance(node, _Or):
        return [list(node.children)]
    return [[node]]


def _literal_to_dimacs(literal, name_to_variable):
    if isinstance(literal, _Atom):
        return name_to_variable[literal.name]
    if isinstance(literal, _Not) and isinstance(literal.child, _Atom):
        return -name_to_variable[literal.child.name]
    raise ValueError(f"Unexpected non-literal node: {type(literal)}")


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_nat_cnf(
    text: str,
    llm_response: bool = False,
    *,
    variable_names: Optional[Sequence[str]] = None,
) -> Tuple[CNFFormula, str, CorrectionStatus]:
    """Parse a natural-language CNF formula into a :class:`CNFFormula`.

    When *llm_response* is ``True``, the content is expected inside
    ``<cnf>…</cnf>`` tags (same ``rfind`` logic as the DIMACS parser).
    When ``variable_names`` is supplied, its order defines the exact DIMACS
    mapping and symbols outside it are rejected.

    Returns ``(formula, error_message, status)``.

    Strict CNF clauses are extracted directly. General boolean formulas are
    distributed to equivalent CNF without auxiliary variables.
    """
    expected_names = normalize_variable_names(variable_names)
    error_message = ""
    preamble = "[ERROR]: "
    status = CorrectionStatus.OK

    if llm_response:
        open_tag = "<cnf>"
        close_tag = "</cnf>"
        end = text.rfind(close_tag)
        start = text.rfind(open_tag, 0, end) if end != -1 else -1
        if start == -1 or end == -1 or start >= end:
            error_message += preamble + "Could not find <cnf>...</cnf> tags in output\n"
            return (
                CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=text),
                error_message,
                CorrectionStatus.INVALID,
            )
        raw_content = text[start + len(open_tag):end].strip()
    else:
        raw_content = text.strip()

    if not raw_content:
        error_message += preamble + "Empty formula: nothing between <cnf>...</cnf> tags\n"
        return (
            CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=text),
            error_message,
            CorrectionStatus.INVALID,
        )

    # Tokenise
    try:
        tokens = _tokenize(raw_content)
    except _NatSyntaxError as exc:
        pattern_key = {
            "invalid_operator": "Invalid operator",
            "parse_error": "Parse error",
            "empty_clause": "Empty clause",
        }.get(exc.code, "Parse error")
        error_message += preamble + f"{pattern_key}: {exc.detail}\n"
        return (
            CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=raw_content),
            error_message,
            CorrectionStatus.INVALID,
        )

    # Parse
    try:
        ast = _Parser(tokens).parse()
    except _NatSyntaxError as exc:
        pattern_key = {
            "empty_clause": "Empty clause",
        }.get(exc.code, "Parse error")
        error_message += preamble + f"{pattern_key}: {exc.detail}\n"
        return (
            CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=raw_content),
            error_message,
            CorrectionStatus.INVALID,
        )

    # Map names to DIMACS indices: the target's order when known, otherwise
    # first appearance.
    var_names = _first_appearance_order(ast)
    if not var_names:
        error_message += preamble + "Parse error: formula contains no variables\n"
        return (
            CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=raw_content),
            error_message,
            CorrectionStatus.INVALID,
        )

    if expected_names is not None:
        expected_set = set(expected_names)
        unknown_names = [name for name in var_names if name not in expected_set]
        if unknown_names:
            rendered = ", ".join(repr(name) for name in unknown_names)
            error_message += preamble + f"Unknown variable: {rendered}\n"
            return (
                CNFFormula(
                    num_vars=len(expected_names),
                    num_clauses=0,
                    clauses=[],
                    raw=raw_content,
                    primary_num_vars=len(expected_names),
                ),
                error_message,
                CorrectionStatus.INVALID,
            )

    mapping_names = expected_names if expected_names is not None else tuple(var_names)
    name_to_variable = {
        name: variable for variable, name in enumerate(mapping_names, start=1)
    }

    is_strict = _is_strict_cnf(ast)
    if not is_strict:
        # General formula: convert to CNF (De Morgan + distribution) without
        # auxiliary variables.
        cnf_ast = _to_nnf(ast)
        try:
            clause_lists = _cnf_distribute(cnf_ast)
        except _NatSyntaxError as exc:
            error_message += preamble + f"Formula too complex: {exc.detail}\n"
            return (
                CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=raw_content),
                error_message,
                CorrectionStatus.INVALID,
            )
    else:
        clause_lists = _strict_clause_lists(ast)

    clauses = [
        [_literal_to_dimacs(literal, name_to_variable) for literal in clause]
        for clause in clause_lists
    ]
    num_vars = len(mapping_names)
    lines = [f"p cnf {num_vars} {len(clauses)}"]
    lines.extend(" ".join(map(str, clause)) + " 0" for clause in clauses)
    formula, err, status = parse_dimacs("\n".join(lines), llm_response=False)
    if formula is not None:
        formula.primary_num_vars = num_vars
        formula.tseitin_converted = not is_strict
    return formula, err, status


# ---------------------------------------------------------------------------
# Public verifier (same signature as dimacs.verify_cnf)
# ---------------------------------------------------------------------------


def verify_nat_cnf(
    text: str,
    llm_response: bool = False,
    *,
    variable_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[CNFFormula], str, CorrectionStatus]:
    """Return ``(formula, error_message, status)`` for a nat CNF string.

    Short-circuits on ``no_cnf_tags`` (same behaviour as
    :func:`~fonesy.evaluation.dimacs.verify_cnf`).
    """
    cnf, error_message, status = parse_nat_cnf(
        text,
        llm_response,
        variable_names=variable_names,
    )
    if status == CorrectionStatus.INVALID:
        return None, error_message, CorrectionStatus.INVALID
    return cnf, error_message, status
