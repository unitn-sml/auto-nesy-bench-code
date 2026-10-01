"""
SymPy program evaluator.

Expected format (inside <cnf>...</cnf> tags):

    from sympy.core.symbol import Symbol
    from sympy.logic.boolalg import to_cnf

    x = Symbol('x')
    y = Symbol('y')

    expr = x | y
    cnf_expr = to_cnf(expr, simplify=False)
    _print_dimacs(cnf_expr)

The program must build a propositional formula out of ``sympy`` ``Symbol``
objects, convert it to CNF with ``sympy.logic.boolalg.to_cnf``, and pass the
result to ``_print_dimacs(cnf_expr)`` — injected automatically before
execution, do not redefine it. ``_print_dimacs`` converts the sympy CNF
expression to a DIMACS formula (via ``to_dimacs_formula`` below) and prints
it to stdout. The last <cnf>...</cnf> block is used (same rfind convention
as DIMACS, PySAT, and CPMpy modes).
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Or, Not

from fonesy.evaluation.dimacs import verify_cnf
from fonesy.evaluation.formula import CNFFormula, normalize_variable_names
from fonesy.statistics.statistics import CorrectionStatus

# ---------------------------------------------------------------------------
# DIMACS conversion helpers. The same source is injected (inspect.getsource)
# before every generated program, see _SYMPY_HELPER.


class DimacsMapping:
    def __init__(self, variable_names=None):
        self._fixed = variable_names is not None
        if self._fixed:
            if isinstance(variable_names, (str, bytes)):
                raise ValueError("variable_names must be a non-string sequence of names")
            names = tuple(variable_names)
            if not names or any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("variable_names must contain non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError("variable_names must contain unique names")
            self._symbol_to_variable = {
                name: variable for variable, name in enumerate(names, start=1)
            }
            self._variable_to_symbol = {
                variable: Symbol(name)
                for variable, name in enumerate(names, start=1)
            }
            self._total_variables = len(names)
        else:
            self._symbol_to_variable = {}
            self._variable_to_symbol = {}
            self._total_variables = 0

    @property
    def total_variables(self):
        return self._total_variables

    def new_variable(self):
        self._total_variables += 1
        return self._total_variables

    def get_variable_for(self, symbol):
        key = symbol.name if self._fixed else symbol
        result = self._symbol_to_variable.get(key)
        if result is None:
            if self._fixed:
                raise ValueError(f"Unknown variable: {symbol.name!r}")
            result = self.new_variable()
            self._symbol_to_variable[key] = result
            self._variable_to_symbol[result] = symbol

        return result

    def get_symbol_for(self, variable):
        return self._variable_to_symbol[variable]

    def __str__(self) -> str:
        return str(self._variable_to_symbol)


class DimacsFormula:
    def __init__(self, mapping, clauses):
        self._mapping = mapping
        self._clauses = clauses

    @property
    def mapping(self):
        return self._mapping

    @property
    def clauses(self):
        return self._clauses

    def __str__(self):
        header = f"p cnf {self._mapping.total_variables} {len(self._clauses)}"
        body = "\n".join(
            " ".join([str(literal) for literal in clause] + ["0"])
            for clause in self._clauses
        )

        return "\n".join([header, body])


def _dimacs_literal(sympy_literal, dimacs_mapping):
    if type(sympy_literal) == Not:
        sympy_symbol, polarity = sympy_literal.args[0], -1
    elif type(sympy_literal) == Symbol:
        sympy_symbol, polarity = sympy_literal, 1
    else:
        raise AssertionError(
            f"invalid cnf: clause literal {sympy_literal!r} (type {type(sympy_literal).__name__}) "
            "is not a Symbol or Not(Symbol). Every clause passed to to_cnf must be built purely "
            "from Boolean Symbol variables combined with &/|/~ -- relational or arithmetic sympy "
            "objects (Eq, Ne, Sum, multiplication of bit-expressions, etc.) cannot appear inside a "
            "CNF clause. Encode arithmetic constraints (e.g. multiplication) with explicit "
            "bit-level Boolean logic (AND/OR/NOT gates for each bit/carry), not symbolic equations."
        )

    dimacs_variable = dimacs_mapping.get_variable_for(sympy_symbol)
    return dimacs_variable * polarity


def to_dimacs_formula(sympy_cnf, variable_names=None):
    """Convert a sympy CNF expression (``to_cnf`` output) to a DimacsFormula.

    ``to_cnf`` collapses a single-clause conjunction to a bare ``Or(...)``,
    and a single-literal formula to a bare ``Symbol``/``Not(...)``, rather
    than wrapping it in ``And(...)``/``Or(...)`` — sympy's n-ary boolean
    operators drop their own wrapper for a single argument, even under
    ``evaluate=False``, so the caller cannot force that shape either. Both
    degenerate cases are normalized here (treated as a 1-clause / 1-literal
    CNF) before applying the same per-literal Not/Symbol -> signed-integer
    logic used for the regular And-of-Or case.
    """
    dimacs_mapping = DimacsMapping(variable_names)
    dimacs_clauses = []

    sympy_clauses = sympy_cnf.args if type(sympy_cnf) == And else [sympy_cnf]
    for sympy_clause in sympy_clauses:
        sympy_literals = (
            sympy_clause.args if type(sympy_clause) == Or else [sympy_clause]
        )
        dimacs_clause = [
            _dimacs_literal(sympy_literal, dimacs_mapping)
            for sympy_literal in sympy_literals
        ]
        dimacs_clauses.append(dimacs_clause)

    return DimacsFormula(dimacs_mapping, dimacs_clauses)


_EXPECTED_VARIABLE_NAMES = None


def _print_dimacs(sympy_cnf) -> None:
    print(str(to_dimacs_formula(sympy_cnf, _EXPECTED_VARIABLE_NAMES)))


# ---------------------------------------------------------------------------
# Code injected before every generated program.

_SYMPY_HELPER = (
    "import json, sys\n"
    "from sympy.core.symbol import Symbol\n"
    "from sympy.logic.boolalg import And, Or, Not, to_cnf\n\n"
    "_EXPECTED_VARIABLE_NAMES = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None\n\n"
    "_EXPECTED_VARIABLE_NAME_SET = (\n"
    "    set(_EXPECTED_VARIABLE_NAMES) if _EXPECTED_VARIABLE_NAMES is not None else None\n"
    ")\n"
    "_ORIGINAL_SYMBOL_NEW = Symbol.__new__\n"
    "def _validated_symbol_new(cls, name, *args, **kwargs):\n"
    "    if (\n"
    "        _EXPECTED_VARIABLE_NAME_SET is not None\n"
    "        and name not in _EXPECTED_VARIABLE_NAME_SET\n"
    "    ):\n"
    "        raise ValueError(f\"Unknown variable: {name!r}\")\n"
    "    return _ORIGINAL_SYMBOL_NEW(cls, name, *args, **kwargs)\n"
    "Symbol.__new__ = staticmethod(_validated_symbol_new)\n\n"
    + inspect.getsource(DimacsMapping) + "\n\n"
    + inspect.getsource(DimacsFormula) + "\n\n"
    + inspect.getsource(_dimacs_literal) + "\n\n"
    + inspect.getsource(to_dimacs_formula) + "\n\n"
    + inspect.getsource(_print_dimacs) + "\n"
)

# ---------------------------------------------------------------------------
# Error-type classification
# ---------------------------------------------------------------------------

_SYMPY_ERROR_PATTERNS: List[Tuple[str, str]] = [
    # sympy-specific errors
    ("no_cnf_tags",             "Could not find <cnf>"),
    ("execution_error",         "Execution error"),
    ("timeout_error",           "execution timed out"),
    ("unknown_variable",        "Unknown variable"),
    # DIMACS output errors (from parse_dimacs on program stdout)
    ("invalid_header",          "Invalid header"),
    ("clause_no_zero",          "Clause must end with 0"),
    ("empty_clause",            "Empty clause"),
    ("invalid_content",         "Invalid DIMACS content"),
    ("missing_num_vars",        "Missing number of variables"),
    ("clause_count_mismatch",   "Clause count mismatch"),
    ("literal_out_of_range",    "exceeds num_vars"),
]

SYMPY_ERROR_TYPE_CODES: List[str] = [code for code, _ in _SYMPY_ERROR_PATTERNS]


def classify_sympy_error_types(error_message: str) -> List[str]:
    """Return sympy error-type codes present in *error_message*."""
    return [code for code, pattern in _SYMPY_ERROR_PATTERNS if pattern in error_message]


_SYMPY_CODE_EXPLANATIONS = {
    "no_cnf_tags":           "missing <cnf>…</cnf> wrapper around the Python program",
    "execution_error":       "the Python program raised an error when executed",
    "timeout_error":         "the Python program exceeded the execution time limit",
    "unknown_variable":      "the formula uses a symbol outside the expected mapping",
    "invalid_header":        "program output: invalid DIMACS header",
    "clause_no_zero":        "program output: clause line does not end with 0",
    "empty_clause":          "program output: empty clause (no literals)",
    "invalid_content":       "program output: non-integer token in clause",
    "missing_num_vars":      "program output: p cnf header is missing num_vars",
    "clause_count_mismatch": "program output: declared clause count does not match actual",
    "literal_out_of_range":  "program output: literal index exceeds num_vars",
}

_SYMPY_CODE_GUIDANCE = {
    "no_cnf_tags": (
        "Wrap your entire Python program in <cnf>...</cnf> tags — "
        "the very first token must be <cnf> and the last must be </cnf>."
    ),
    "execution_error": (
        "Fix the Python error shown above. Make sure Symbol/to_cnf are imported, "
        "the formula is built correctly, and to_cnf's result is passed to "
        "_print_dimacs unchanged."
    ),
    "timeout_error": (
        "The program took too long. Avoid infinite loops and reduce computation — "
        "converting a small formula to CNF should take well under a second."
    ),
    "unknown_variable": (
        "Use only Symbol names from the ordered variable mapping in the prompt."
    ),
    "invalid_header": (
        "Ensure _print_dimacs produces a valid 'p cnf <num_vars> <num_clauses>' header."
    ),
    "clause_no_zero": (
        "Every clause line printed to stdout must end with ' 0' — "
        "_print_dimacs handles this automatically; do not suppress its output."
    ),
    "empty_clause": (
        "Every clause must contain at least one literal. "
        "Make sure the formula passed to to_cnf is not vacuously empty."
    ),
    "invalid_content": (
        "Only integers are valid in clause lines. "
        "Do not add extra print statements or comments to stdout."
    ),
    "missing_num_vars": (
        "The 'p cnf' header must declare the number of variables and clauses — "
        "_print_dimacs handles this automatically; do not suppress its output."
    ),
    "clause_count_mismatch": (
        "The declared clause count must equal the number of clause lines — "
        "do not add extra print statements that corrupt stdout."
    ),
    "literal_out_of_range": (
        "All literal indices must be in the range [1, num_vars]. "
        "Check that Symbol names are used consistently."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXECUTION_TIMEOUT = 600  # seconds


def _extract_code(text: str) -> Optional[str]:
    """Return the content of the last <cnf>…</cnf> block, or None if not found."""
    open_tag = "<cnf>"
    close_tag = "</cnf>"
    end = text.rfind(close_tag)
    if end == -1:
        return None
    start = text.rfind(open_tag, 0, end)
    if start == -1 or start >= end:
        return None
    return text[start + len(open_tag): end].strip()


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_sympy_cnf(
    text: str,
    llm_response: bool = True,
    *,
    variable_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[CNFFormula], str, CorrectionStatus]:
    """Parse and execute a SymPy program from *text*, then verify its DIMACS output.

    Args:
        text: Raw LLM output (or bare Python code when *llm_response* is ``False``).
        llm_response: When ``True`` the program is expected inside ``<cnf>…</cnf>``
            tags (last block wins). When ``False``, *text* is treated as raw Python.
        variable_names: Ordered target mapping; each ``Symbol`` name must be
            present and maps to its one-based position.

    Returns:
        ``(formula, error_message, status)`` — the same triple returned by
        :func:`fonesy.evaluation.dimacs.parse_dimacs`.
    """
    expected_names = normalize_variable_names(variable_names)

    if llm_response:
        code = _extract_code(text)
        if code is None:
            return (
                None,
                "[ERROR]: Could not find <cnf>...</cnf> tags in output\n",
                CorrectionStatus.INVALID,
            )
    else:
        code = text.strip()

    full_code = _SYMPY_HELPER + code
    command = [sys.executable, "-c", full_code]
    if expected_names is not None:
        command.append(json.dumps(expected_names))

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_EXECUTION_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return (
            None,
            "[ERROR]: Program execution timed out\n",
            CorrectionStatus.INVALID,
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Unknown variable:" in stderr:
            detail = next(
                line.split("Unknown variable:", 1)[1]
                for line in reversed(stderr.splitlines())
                if "Unknown variable:" in line
            )
            return (
                None,
                f"[ERROR]: Unknown variable:{detail}\n",
                CorrectionStatus.INVALID,
            )
        return (
            None,
            f"[ERROR]: Execution error\n{stderr}\n",
            CorrectionStatus.INVALID,
        )

    stdout = result.stdout.strip()
    if not stdout:
        return (
            None,
            "[ERROR]: Invalid DIMACS content: program printed no output\n",
            CorrectionStatus.INVALID,
        )

    try:
        return verify_cnf(
            stdout,
            llm_response=False,
            primary_num_vars=(len(expected_names) if expected_names is not None else None),
        )
    except Exception as exc:
        return (
            None,
            f"[ERROR]: Invalid DIMACS content: {exc}\n",
            CorrectionStatus.INVALID,
        )



# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def try_import_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Prepend missing SymPy imports to the program inside <cnf> tags.

    Returns the fixed text (with tags) when at least one import was added,
    or ``None`` when both imports were already present (nothing to fix).
    """
    if llm_response:
        code = _extract_code(text)
        if code is None:
            return None
    else:
        code = text.strip()

    needs_symbol = "from sympy.core.symbol import Symbol" not in code
    needs_to_cnf = "from sympy.logic.boolalg import to_cnf" not in code

    if not needs_symbol and not needs_to_cnf:
        return None

    prepend_lines = []
    if needs_symbol:
        prepend_lines.append("from sympy.core.symbol import Symbol")
    if needs_to_cnf:
        prepend_lines.append("from sympy.logic.boolalg import to_cnf")

    fixed_code = "\n".join(prepend_lines) + "\n" + code

    if llm_response:
        return f"<cnf>\n{fixed_code}\n</cnf>"
    return fixed_code
