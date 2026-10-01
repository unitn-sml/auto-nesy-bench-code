"""
CPMpy program evaluator.

Expected format (inside <cnf>...</cnf> tags):

    import cpmpy as cp

    x = cp.boolvar(name='x')
    y = cp.boolvar(name='y')

    _print_dimacs([x | y])

The program must import cpmpy itself (``import cpmpy as cp``). The helper
``_print_dimacs(constraints)`` is injected automatically before execution —
do not redefine it.

``_print_dimacs`` calls ``cpmpy.transformations.to_cnf.to_cnf`` (backed by
pindakaas) and then bridges through the CPM_pysat solver to produce a DIMACS
formula on stdout.  The last <cnf>...</cnf> block is used (same rfind convention
as DIMACS and PySAT modes).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

from fonesy.evaluation.dimacs import verify_cnf
from fonesy.evaluation.formula import CNFFormula, normalize_variable_names
from fonesy.statistics.statistics import CorrectionStatus

# ---------------------------------------------------------------------------
# Helper boilerplate injected before every LLM program
# ---------------------------------------------------------------------------

_CPMPY_HELPER = """\
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.solvers.pysat import CPM_pysat
from pysat.formula import CNF
import json, sys

_EXPECTED_VARIABLE_NAMES = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None

def _print_dimacs(constraints):
    if not isinstance(constraints, list):
        constraints = list(constraints)
    if _EXPECTED_VARIABLE_NAMES is not None:
        expected = set(_EXPECTED_VARIABLE_NAMES)
        for variable in get_variables(constraints):
            if not variable.is_bool():
                raise ValueError(f"Non-Boolean variable: {variable.name!r}")
            if variable.name not in expected:
                raise ValueError(f"Unknown variable: {variable.name!r}")
    cnf_cs = to_cnf(constraints)
    solver = CPM_pysat()
    if _EXPECTED_VARIABLE_NAMES is not None:
        for name in _EXPECTED_VARIABLE_NAMES:
            solver.pysat_vpool.id(name)
    captured = []
    class _CS:
        def __init__(self, r): self._r = r
        def add_clause(self, c, *a, **k): captured.append(list(c)); return self._r.add_clause(c, *a, **k)
        def append_formula(self, f, *a, **k): captured.extend([list(c) for c in f]); return self._r.append_formula(f, *a, **k)
        def __getattr__(self, n): return getattr(self._r, n)
    solver.pysat_solver = _CS(solver.pysat_solver)
    for c in solver.transform(cnf_cs):
        solver._post_constraint(c)
    n = max(solver.pysat_vpool.id2obj) if solver.pysat_vpool.id2obj else 0
    cnf = CNF(from_clauses=captured)
    cnf.nv = n
    cnf.to_fp(sys.stdout)
"""

# ---------------------------------------------------------------------------
# Error-type classification
# ---------------------------------------------------------------------------

_CPMPY_ERROR_PATTERNS: List[Tuple[str, str]] = [
    ("no_cnf_tags",             "Could not find <cnf>"),
    ("execution_error",         "Execution error"),
    ("timeout_error",           "execution timed out"),
    ("unknown_variable",        "Unknown variable"),
    ("non_boolean_variable",    "Non-Boolean variable"),
    ("invalid_header",          "Invalid header"),
    ("clause_no_zero",          "Clause must end with 0"),
    ("empty_clause",            "Empty clause"),
    ("invalid_content",         "Invalid DIMACS content"),
    ("missing_num_vars",        "Missing number of variables"),
    ("clause_count_mismatch",   "Clause count mismatch"),
    ("literal_out_of_range",    "exceeds num_vars"),
]

CPMPY_ERROR_TYPE_CODES: List[str] = [code for code, _ in _CPMPY_ERROR_PATTERNS]


def classify_cpmpy_error_types(error_message: str) -> List[str]:
    """Return CPMpy error-type codes present in *error_message*."""
    return [code for code, pattern in _CPMPY_ERROR_PATTERNS if pattern in error_message]


_CPMPY_CODE_EXPLANATIONS = {
    "no_cnf_tags":           "missing <cnf>…</cnf> wrapper around the CPMpy program",
    "execution_error":       "the CPMpy program raised an error when executed",
    "timeout_error":         "the CPMpy program exceeded the execution time limit",
    "unknown_variable":      "the constraints use a name outside the expected mapping",
    "non_boolean_variable":  "the constraints use a non-Boolean decision variable",
    "invalid_header":        "program output: invalid DIMACS header",
    "clause_no_zero":        "program output: clause line does not end with 0",
    "empty_clause":          "program output: empty clause (no literals)",
    "invalid_content":       "program output: non-integer token in clause",
    "missing_num_vars":      "program output: p cnf header is missing num_vars",
    "clause_count_mismatch": "program output: declared clause count does not match actual",
    "literal_out_of_range":  "program output: literal index exceeds num_vars",
}

_CPMPY_CODE_GUIDANCE = {
    "no_cnf_tags": (
        "Wrap your entire CPMpy program in <cnf>...</cnf> tags — "
        "the very first token must be <cnf> and the last must be </cnf>."
    ),
    "execution_error": (
        "Fix the Python error shown above.  Make sure variable names are defined "
        "and the constraint list passed to _print_dimacs is valid."
    ),
    "timeout_error": (
        "The program took too long.  Avoid infinite loops and reduce computation — "
        "building a CPMpy model should take well under a second."
    ),
    "unknown_variable": (
        "Use only Boolean variable names from the ordered mapping in the prompt."
    ),
    "non_boolean_variable": (
        "Use cp.boolvar variables only; integer decision variables are not valid CNF symbols."
    ),
    "invalid_header": (
        "Ensure _print_dimacs produces a valid 'p cnf <num_vars> <num_clauses>' header."
    ),
    "clause_no_zero": (
        "Every clause line printed to stdout must end with ' 0' — "
        "call _print_dimacs correctly so it handles formatting."
    ),
    "empty_clause": (
        "Every clause must contain at least one literal.  "
        "Avoid passing empty constraint lists to _print_dimacs."
    ),
    "invalid_content": (
        "Only integers are valid in clause lines.  "
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
        "All literal indices must be in the range [1, num_vars].  "
        "Check that variable names match the required mapping."
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


def verify_cpmpy_cnf(
    text: str,
    llm_response: bool = True,
    *,
    variable_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[CNFFormula], str, CorrectionStatus]:
    """Parse and execute a CPMpy program from *text*, then verify its DIMACS output.

    Args:
        text: Raw LLM output (or bare Python code when *llm_response* is ``False``).
        llm_response: When ``True`` the program is expected inside ``<cnf>…</cnf>``
            tags (last block wins).  When ``False``, *text* is treated as raw Python.
        variable_names: Ordered target mapping used to seed CPMpy's PySAT
            variable pool before transformations introduce auxiliaries.

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

    full_code = _CPMPY_HELPER + code
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
        for marker in ("Unknown variable:", "Non-Boolean variable:"):
            if marker in stderr:
                detail = next(
                    line.split(marker, 1)[1]
                    for line in reversed(stderr.splitlines())
                    if marker in line
                )
                return (
                    None,
                    f"[ERROR]: {marker}{detail}\n",
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


class _BoolvarNameTransformer(ast.NodeTransformer):
    """Rewrite unambiguous positional CPMpy Boolean-variable names."""

    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        is_cp_boolvar = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "boolvar"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cp"
        )
        if not is_cp_boolvar or len(node.args) != 1:
            return node

        positional_name = node.args[0]
        is_string_constant = (
            isinstance(positional_name, ast.Constant)
            and isinstance(positional_name.value, str)
        )
        # An f-string first argument is also a name.
        is_fstring = isinstance(positional_name, ast.JoinedStr)
        if not (is_string_constant or is_fstring):
            return node

        # Calls with name= or **kwargs are left to the verifier.
        if any(keyword.arg in (None, "name") for keyword in node.keywords):
            return node

        node.args.clear()
        node.keywords.append(ast.keyword(arg="name", value=positional_name))
        self.changed = True
        return node


def try_boolvar_name_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Convert ``cp.boolvar('x')`` to ``cp.boolvar(name='x')``.

    Only exact ``cp.boolvar`` calls with one positional string argument, no
    existing ``name=`` keyword, and no ``**kwargs`` are changed. The whole
    program is parsed as Python so comments and string contents cannot be
    mistaken for calls. Returns ``None`` when the input is not applicable or
    cannot be parsed; the caller remains responsible for verifying the fixed
    program before accepting it.
    """
    if llm_response:
        code = _extract_code(text)
        if code is None:
            return None
    else:
        code = text.strip()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    transformer = _BoolvarNameTransformer()
    fixed_tree = transformer.visit(tree)
    if not transformer.changed:
        return None

    ast.fix_missing_locations(fixed_tree)
    fixed_code = ast.unparse(fixed_tree)
    if llm_response:
        return f"<cnf>\n{fixed_code}\n</cnf>"
    return fixed_code


def try_import_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Prepend a missing ``import cpmpy as cp`` to the program inside <cnf> tags.

    Returns the fixed text (with tags) when the import was added,
    or ``None`` when the import was already present (nothing to fix).
    """
    if llm_response:
        code = _extract_code(text)
        if code is None:
            return None
    else:
        code = text.strip()

    if "import cpmpy" in code:
        return None

    fixed_code = "import cpmpy as cp\n" + code

    if llm_response:
        return f"<cnf>\n{fixed_code}\n</cnf>"
    return fixed_code
