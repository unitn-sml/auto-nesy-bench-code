"""
PySAT program evaluator.

Expected format (inside <cnf>...</cnf> tags):

    from pysat.formula import CNF
    import sys

    cnf = CNF()
    cnf.append([1, 2])
    cnf.append([-1, 3])
    cnf.to_fp(sys.stdout)

The program must print a valid DIMACS formula to stdout when executed.
The last <cnf>...</cnf> block is used (same rfind convention as DIMACS).
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import List, Optional, Tuple

from fonesy.evaluation.dimacs import verify_cnf
from fonesy.evaluation.formula import CNFFormula, validate_primary_num_vars
from fonesy.statistics.statistics import CorrectionStatus

# ---------------------------------------------------------------------------
# Error-type classification
# ---------------------------------------------------------------------------

_PYSAT_ERROR_PATTERNS: List[Tuple[str, str]] = [
    # pysat-specific errors
    ("no_cnf_tags",             "Could not find <cnf>"),
    ("execution_error",         "Execution error"),
    ("timeout_error",           "execution timed out"),
    ("encoder_aux_collision",   "PySAT encoder auxiliary-variable collision risk"),
    # DIMACS output errors (from parse_dimacs on program stdout)
    ("invalid_header",          "Invalid header"),
    ("clause_no_zero",          "Clause must end with 0"),
    ("empty_clause",            "Empty clause"),
    ("invalid_content",         "Invalid DIMACS content"),
    ("missing_num_vars",        "Missing number of variables"),
    ("clause_count_mismatch",   "Clause count mismatch"),
    ("literal_out_of_range",    "exceeds num_vars"),
]

PYSAT_ERROR_TYPE_CODES: List[str] = [code for code, _ in _PYSAT_ERROR_PATTERNS]


def classify_pysat_error_types(error_message: str) -> List[str]:
    """Return pysat error-type codes present in *error_message*."""
    return [code for code, pattern in _PYSAT_ERROR_PATTERNS if pattern in error_message]


_PYSAT_CODE_EXPLANATIONS = {
    "no_cnf_tags":           "missing <cnf>…</cnf> wrapper around the Python program",
    "execution_error":       "the Python program raised an error when executed",
    "timeout_error":         "the Python program exceeded the execution time limit",
    "encoder_aux_collision": (
        "a PySAT cardinality or pseudo-Boolean encoder was not told where the "
        "target's reserved primary-variable range ends"
    ),
    "invalid_header":        "program output: invalid DIMACS header",
    "clause_no_zero":        "program output: clause line does not end with 0",
    "empty_clause":          "program output: empty clause (no literals)",
    "invalid_content":       "program output: non-integer token in clause",
    "missing_num_vars":      "program output: p cnf header is missing num_vars",
    "clause_count_mismatch": "program output: declared clause count does not match actual",
    "literal_out_of_range":  "program output: literal index exceeds num_vars",
}

_PYSAT_CODE_GUIDANCE = {
    "no_cnf_tags": (
        "Wrap your entire Python program in <cnf>...</cnf> tags — "
        "the very first token must be <cnf> and the last must be </cnf>."
    ),
    "execution_error": (
        "Fix the Python error shown above.  Make sure all imports are correct, "
        "variable names are defined, and the logic produces a valid CNF object."
    ),
    "timeout_error": (
        "The program took too long.  Avoid infinite loops and reduce computation — "
        "building a CNF should take well under a second."
    ),
    "encoder_aux_collision": (
        "Pass top_id=N to every CardEnc/PBEnc call, where N is the largest "
        "primary variable ID, or share an IDPool(start_from=N+1) through its "
        "vpool argument. This ensures every generated auxiliary ID is above N."
    ),
    "invalid_header": (
        "Ensure the first printed line is 'p cnf <num_vars> <num_clauses>' with correct integers."
    ),
    "clause_no_zero": (
        "Every clause line printed to stdout must end with ' 0'."
    ),
    "empty_clause": (
        "Every clause must contain at least one literal.  Remove empty append() calls."
    ),
    "invalid_content": (
        "Only integers are valid in clause lines.  Do not print comments or extra text."
    ),
    "missing_num_vars": (
        "The 'p cnf' header must declare the number of variables and clauses."
    ),
    "clause_count_mismatch": (
        "The declared clause count in the header must equal the number of clause lines printed."
    ),
    "literal_out_of_range": (
        "All literal indices must be in the range [1, num_vars]."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXECUTION_TIMEOUT = 600  # seconds
_ENCODER_RESERVATION_MARKER = "FONESY_PYSAT_ENCODER_RESERVATION:"


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


def _guard_encoder_auxiliaries(code: str, primary_num_vars: int) -> str:
    """Wrap prediction code with guards around PySAT's auxiliary encoders.

    ``CardEnc`` and ``PBEnc`` otherwise infer their first auxiliary ID from the
    literals in an individual constraint.  If that constraint does not mention
    every target variable, the inferred ID can fall inside the target's
    reserved primary range.  When neither mechanism is supplied, the runtime
    wrapper transparently defaults ``top_id`` to the authoritative primary
    count. Explicit reservations are validated. Wrapping at runtime covers
    aliases, positional calls, keyword calls, and imports performed inside
    helper functions.

    The user program is compiled separately so a leading ``from __future__``
    import remains valid.
    """
    prologue = f"""
import inspect as _fonesy_inspect

_FONESY_PRIMARY_NUM_VARS = {primary_num_vars!r}
_FONESY_RESERVATION_MARKER = {_ENCODER_RESERVATION_MARKER!r}

def _fonesy_install_encoder_guards(_encoder_cls, _encoder_label, _method_names):
    def _fonesy_make_guard(_original, _signature, _method_name):
        def _fonesy_guarded(_runtime_cls, *args, **kwargs):
            _bound = _signature.bind(*args, **kwargs)
            _bound.apply_defaults()
            _top_id = _bound.arguments.get('top_id')
            _vpool = _bound.arguments.get('vpool')
            if _top_id is None and _vpool is None:
                # Preserve the encoder's semantics while moving any generated
                # namespace above the complete target primary universe.
                _bound.arguments['top_id'] = _FONESY_PRIMARY_NUM_VARS
                _top_id = _FONESY_PRIMARY_NUM_VARS
            _reserved_top_id = False
            if _top_id is not None:
                try:
                    _reserved_top_id = _top_id >= _FONESY_PRIMARY_NUM_VARS
                except TypeError:
                    _reserved_top_id = False
            _reserved_vpool = False
            if _vpool is not None:
                try:
                    _reserved_vpool = _vpool.top >= _FONESY_PRIMARY_NUM_VARS
                except (AttributeError, TypeError):
                    _reserved_vpool = False

            if not (_reserved_top_id or _reserved_vpool):
                raise ValueError(
                    _FONESY_RESERVATION_MARKER
                    + f'{{_encoder_label}}.{{_method_name}} must receive '
                    + f'top_id >= {{_FONESY_PRIMARY_NUM_VARS}} or an '
                    + f'IDPool with top >= {{_FONESY_PRIMARY_NUM_VARS}} '
                    + f'(construct it with IDPool(start_from='
                    + f'{{_FONESY_PRIMARY_NUM_VARS + 1}}))'
                )

            return _original(*_bound.args, **_bound.kwargs)

        return classmethod(_fonesy_guarded)

    for _method_name in _method_names:
        _original = getattr(_encoder_cls, _method_name)
        _signature = _fonesy_inspect.signature(_original)
        setattr(
            _encoder_cls,
            _method_name,
            _fonesy_make_guard(_original, _signature, _method_name),
        )

from pysat.card import CardEnc as _FonesyCardEnc
_fonesy_install_encoder_guards(
    _FonesyCardEnc,
    'CardEnc',
    ('atmost', 'atleast', 'equals'),
)

try:
    from pysat.pb import PBEnc as _FonesyPBEnc
except Exception:
    _FonesyPBEnc = None

if _FonesyPBEnc is not None:
    _fonesy_install_encoder_guards(
        _FonesyPBEnc,
        'PBEnc',
        ('_encode', 'leq', 'atmost', 'geq', 'atleast', 'equals'),
    )
"""
    return (
        prologue
        + "\n_fonesy_prediction_globals = {'__name__': '__main__'}\n"
        + f"exec(compile({code!r}, '<pysat-prediction>', 'exec'), "
        + "_fonesy_prediction_globals)\n"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_pysat_cnf(
    text: str,
    llm_response: bool = True,
    *,
    primary_num_vars: Optional[int] = None,
) -> Tuple[Optional[CNFFormula], str, CorrectionStatus]:
    """Parse and execute a PySAT program from *text*, then verify its DIMACS output.

    Args:
        text: Raw LLM output (or bare Python code when *llm_response* is ``False``).
        llm_response: When ``True`` the program is expected inside ``<cnf>…</cnf>``
            tags (last block wins).  When ``False``, *text* is treated as raw Python.
        primary_num_vars: Target-defined primary variable count. Referenced
            higher IDs are canonicalized as existential auxiliaries.

    Returns:
        ``(formula, error_message, status)`` — the same triple returned by
        :func:`fonesy.evaluation.dimacs.parse_dimacs`.
    """
    primary_num_vars = validate_primary_num_vars(primary_num_vars)

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

    execution_code = code
    if primary_num_vars is not None:
        execution_code = _guard_encoder_auxiliaries(code, primary_num_vars)

    # Execute in a subprocess so it cannot corrupt our process state.
    try:
        result = subprocess.run(
            [sys.executable, "-c", execution_code],
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
        reservation_error = next(
            (
                line.split(_ENCODER_RESERVATION_MARKER, 1)[1].strip()
                for line in stderr.splitlines()
                if _ENCODER_RESERVATION_MARKER in line
            ),
            None,
        )
        if reservation_error is not None:
            return (
                None,
                "[ERROR]: PySAT encoder auxiliary-variable collision risk: "
                f"{reservation_error}\n",
                CorrectionStatus.INVALID,
            )
        return (
            None,
            f"[ERROR]: Execution error\n{stderr}\n",
            CorrectionStatus.INVALID,
        )

    # Delegate DIMACS validation to the existing parser.
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
            primary_num_vars=primary_num_vars,
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


# `from pysat import CNF` (a common LLM mistake) is removed and replaced by
# the correct import.
_BAD_PYSAT_TOPLEVEL_IMPORT_RE = re.compile(r"^[ \t]*from[ \t]+pysat[ \t]+import[ \t]+.*$", re.MULTILINE)


def try_import_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Fix missing/incorrect PySAT/sys imports in the program inside <cnf> tags.

    Strips any bad ``from pysat import ...`` line (``CNF`` is only importable
    from ``pysat.formula``) and prepends the correct imports if missing.
    Returns the fixed text (with tags) when something was changed, or
    ``None`` when the imports were already correct (nothing to fix).
    """
    if llm_response:
        code = _extract_code(text)
        if code is None:
            return None
    else:
        code = text.strip()

    stripped_code = _BAD_PYSAT_TOPLEVEL_IMPORT_RE.sub("", code)
    removed_bad_import = stripped_code != code
    code = stripped_code

    needs_cnf = "from pysat.formula import CNF" not in code
    needs_sys = "import sys" not in code

    if not needs_cnf and not needs_sys and not removed_bad_import:
        return None

    prepend_lines = []
    if needs_cnf:
        prepend_lines.append("from pysat.formula import CNF")
    if needs_sys:
        prepend_lines.append("import sys")

    fixed_code = "\n".join(prepend_lines) + "\n" + code if prepend_lines else code

    if llm_response:
        return f"<cnf>\n{fixed_code}\n</cnf>"
    return fixed_code
