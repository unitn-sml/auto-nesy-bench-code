"""
DIMACS CNF parser with validation.
"""

from typing import List, Tuple, Optional
from fonesy.evaluation.formula import CNFFormula, validate_primary_num_vars
from fonesy.statistics.statistics import CorrectionStatus

#: Error codes that indicate the preamble (``p cnf`` header) is wrong but the
#: clause bodies are syntactically correct.  These can be auto-fixed by
#: recomputing the header from the actual clauses without querying the LLM.
PREAMBLE_ERROR_CODES: frozenset = frozenset(
    {
        "missing_num_vars",
        "clause_count_mismatch",
        "literal_out_of_range",
        "duplicate_header",
        "clause_before_header",
        "negative_declared_count",
    }
)

#: Error codes that indicate clause bodies are missing the trailing ``0``
#: terminator.  These can be auto-fixed by appending `` 0`` to each offending
#: line without querying the LLM.
CLAUSE_ERROR_CODES: frozenset = frozenset({"clause_no_zero"})


# ---------------------------------------------------------------------------
# Error-type classification
# ---------------------------------------------------------------------------

#: Short codes for every distinct error that ``parse_dimacs`` or ``is_valid``
#: can emit, paired with the substring that uniquely identifies them in the
#: error message produced by those functions.
_ERROR_PATTERNS: List[Tuple[str, str]] = [
    ("no_cnf_tags", "Could not find <cnf>"),
    ("invalid_header", "Invalid header"),
    ("clause_no_zero", "Clause must end with 0"),
    ("empty_clause", "Empty clause"),
    ("invalid_content", "Invalid DIMACS content"),
    ("missing_num_vars", "Missing number of variables"),
    ("clause_count_mismatch", "Clause count mismatch"),
    ("literal_out_of_range", "exceeds num_vars"),
    ("duplicate_header", "Duplicate header"),
    ("clause_before_header", "Clause before header"),
    ("negative_declared_count", "Negative declared"),
    ("zero_literal_in_clause", "contains literal 0"),
]

#: All recognised error-type codes, in a stable order.
ERROR_TYPE_CODES: List[str] = [code for code, _ in _ERROR_PATTERNS]


def _extract_cnf_block(text: str) -> Optional[str]:
    """Return the content of the last well-formed <cnf>...</cnf> block in
    text, or None if no such block exists.

    Uses rfind so that when a reasoning model emits multiple <cnf> blocks
    (e.g. a draft followed by a corrected version), the last one — the
    model's final answer — is always the one parsed.
    """
    open_tag, close_tag = "<cnf>", "</cnf>"
    end = text.rfind(close_tag)
    start = text.rfind(open_tag, 0, end) if end != -1 else -1
    if start == -1 or end == -1 or start >= end:
        return None
    return text[start + len(open_tag) : end].strip()


def classify_error_types(error_message: str) -> List[str]:
    """Return the list of error-type codes present in *error_message*.

    The function scans the message produced by :func:`parse_dimacs` /
    :meth:`~fonesy.evaluation.formula.CNFFormula.is_valid` for known
    substrings and maps each hit to a short, stable code.  Multiple codes
    are returned when the message contains more than one error.

    Args:
        error_message: The raw error string accumulated during parsing /
            validation.  An empty string or a message with no known
            patterns returns an empty list.

    Returns:
        Deduplicated list of error-type codes in the order they appear in
        :data:`ERROR_TYPE_CODES`.

    Example::

        >>> classify_error_types("[ERROR]: Clause must end with 0 (line 3)\\n")
        ['clause_no_zero']
        >>> classify_error_types("")
        []
    """
    return [code for code, pattern in _ERROR_PATTERNS if pattern in error_message]


def canonicalize_primary_variables(
    cnf: CNFFormula,
    primary_num_vars: int,
) -> CNFFormula:
    """Return *cnf* in the target's primary-variable coordinate system.

    Variables ``1..primary_num_vars`` retain their IDs.  Referenced variables
    above that range are prediction-local auxiliaries and are densely renamed
    in ascending original-ID order.  Header-only extra variables disappear,
    while omitted primary variables remain declared and therefore free.

    The input is expected to have passed :meth:`CNFFormula.is_valid` first.
    """
    primary_num_vars = validate_primary_num_vars(primary_num_vars)
    assert primary_num_vars is not None

    auxiliary_ids = sorted(
        {
            abs(literal)
            for clause in cnf.clauses
            for literal in clause
            if abs(literal) > primary_num_vars
        }
    )
    auxiliary_map = {
        old_id: primary_num_vars + offset
        for offset, old_id in enumerate(auxiliary_ids, start=1)
    }

    clauses = [
        [
            (-1 if literal < 0 else 1)
            * auxiliary_map.get(abs(literal), abs(literal))
            for literal in clause
        ]
        for clause in cnf.clauses
    ]
    num_vars = primary_num_vars + len(auxiliary_ids)
    lines = [f"p cnf {num_vars} {len(clauses)}"]
    lines.extend(" ".join(map(str, clause)) + " 0" for clause in clauses)
    return CNFFormula(
        num_vars=num_vars,
        num_clauses=len(clauses),
        clauses=clauses,
        raw="\n".join(lines),
        primary_num_vars=primary_num_vars,
    )


def verify_cnf(
    text: str,
    llm_response: bool = False,
    *,
    primary_num_vars: Optional[int] = None,
) -> Tuple[Optional[CNFFormula], str, CorrectionStatus]:
    """Return ``(formula, error_message, status)`` for *text*.

    When ``primary_num_vars`` is supplied, valid predictions are canonicalized
    into that primary universe and referenced higher IDs become dense
    prediction-local auxiliaries.

    When ``no_cnf_tags`` is the error, the function returns immediately without
    calling :meth:`~fonesy.evaluation.formula.CNFFormula.is_valid` — the formula
    skeleton is empty and ``is_valid`` would otherwise add spurious
    ``missing_num_vars`` / ``clause_count_mismatch`` errors that would mislead
    the LLM correction prompt.  All other error types continue to accumulate
    additional diagnostics from ``is_valid`` as before.
    """
    primary_num_vars = validate_primary_num_vars(primary_num_vars)
    cnf, error_message, status = parse_dimacs(text, llm_response)
    if "no_cnf_tags" in classify_error_types(error_message):
        return None, error_message, CorrectionStatus.INVALID
    valid, error_message = cnf.is_valid(error_message)
    if valid == CorrectionStatus.INVALID or status == CorrectionStatus.INVALID:
        return None, error_message, CorrectionStatus.INVALID
    if primary_num_vars is not None:
        cnf = canonicalize_primary_variables(cnf, primary_num_vars)
    return cnf, error_message, status


def parse_dimacs(text: str, llm_response: bool = False) -> CNFFormula:
    """
    Parses DIMACS CNF string into CNFFormula object.
    If llm_response=True, expects the DIMACS content to be wrapped in
    <cnf></cnf> tags and extracts only that portion before parsing.
    """
    error_message = ""
    preamble = "[ERROR]: "
    status = CorrectionStatus.OK
    clauses = []
    num_vars = None
    num_clauses = None

    if llm_response:
        dimacs = _extract_cnf_block(text)
        if dimacs is None:
            status = CorrectionStatus.INVALID
            error_message += preamble + "Could not find <cnf>...</cnf> tags in output\n"
            return (
                CNFFormula(
                    num_vars=num_vars,
                    num_clauses=num_clauses,
                    clauses=clauses,
                    raw=text,
                ),
                error_message,
                status,
            )
    else:
        dimacs = text.strip()

    # A duplicate header or a clause before the header is left to `is_valid`,
    # so that `try_preamble_fix` can still repair it.
    header_seen = False
    duplicate_header = False
    clause_before_header = False

    try:
        lines = dimacs.splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                if header_seen:
                    duplicate_header = True
                header_seen = True
                parts = line.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    status = CorrectionStatus.INVALID
                    error_message += preamble + f"Invalid header at line {idx}\n"
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            else:
                if not header_seen:
                    clause_before_header = True
                literals = list(map(int, line.split()))
                if literals[-1] != 0:
                    status = CorrectionStatus.INVALID
                    error_message += preamble + f"Clause must end with 0 (line {idx})\n"
                clause = literals[:-1]
                if not clause:
                    status = CorrectionStatus.INVALID
                    error_message += preamble + f"Empty clause at line {idx}\n"
                clauses.append(clause)
    except Exception as e:
        if llm_response:
            status = CorrectionStatus.INVALID
            error_message += preamble + f"Invalid DIMACS content: {e}\n"
            return (
                CNFFormula(
                    num_vars=num_vars,
                    num_clauses=num_clauses,
                    clauses=clauses,
                    raw=dimacs,
                ),
                error_message,
                status,
            )
        else:
            raise ValueError(f"Invalid target")

    cnf = CNFFormula(
        num_vars=num_vars,
        num_clauses=num_clauses,
        clauses=clauses,
        raw=dimacs,  # store only the extracted DIMACS, not the full raw text
        duplicate_header=duplicate_header,
        clause_before_header=clause_before_header,
    )
    return cnf, error_message, status


def try_preamble_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Attempt to fix a DIMACS string whose only errors are in the preamble.

    When the clause bodies are syntactically correct but the ``p cnf X Y``
    header contains wrong values (e.g. wrong clause count, missing num_vars, or
    a num_vars that is smaller than the largest literal used), the correct header
    can be recomputed directly from the clauses without querying the LLM.

    Args:
        text: Raw text to fix.  When *llm_response* is ``True`` the DIMACS
            content is expected inside ``<cnf>…</cnf>`` tags, and the returned
            string is also wrapped in those tags.
        llm_response: Whether *text* comes from an LLM response (with tags).

    Returns:
        A corrected DIMACS string (with tags when *llm_response* is ``True``),
        or ``None`` if the errors are not exclusively preamble-related (e.g.
        clause-level syntax errors or missing tags).
    """
    cnf, error_message, parse_status = parse_dimacs(text, llm_response)
    if parse_status == CorrectionStatus.INVALID:
        return None

    valid, full_error = cnf.is_valid(error_message)
    if valid != CorrectionStatus.INVALID:
        return None

    codes = classify_error_types(full_error)
    if not codes or not all(c in PREAMBLE_ERROR_CODES for c in codes):
        return None

    if not cnf.clauses:
        return None

    all_lits = [abs(lit) for clause in cnf.clauses for lit in clause]
    if not all_lits:
        return None

    num_vars = max(all_lits)
    num_clauses = len(cnf.clauses)
    lines = [f"p cnf {num_vars} {num_clauses}"]
    for clause in cnf.clauses:
        lines.append(" ".join(map(str, clause)) + " 0")
    raw = "\n".join(lines)

    return f"<cnf>\n{raw}\n</cnf>" if llm_response else raw


def try_clause_fix(text: str, llm_response: bool = True) -> Optional[str]:
    """Attempt to fix clauses that are missing the trailing ``0`` terminator.

    **Sequential two-pass design** — this function is *step 1* of a two-step
    repair chain and must always be called before :func:`try_preamble_fix`:

    1. :func:`try_clause_fix` appends the missing `` 0`` terminator to every
       offending clause line.  It accepts formulas whose error set is a subset
       of ``CLAUSE_ERROR_CODES | PREAMBLE_ERROR_CODES``, meaning it handles
       the mixed case where ``clause_no_zero`` and preamble errors (e.g.
       ``clause_count_mismatch``) coexist.  After the clause pass it calls
       :func:`try_preamble_fix` internally to clean up any residual header
       errors introduced or exposed by the clause fix.
    2. :func:`try_preamble_fix` (*step 2*, called internally at the end) then
       recomputes the ``p cnf`` header from the now-correct clause list.

    Callers should therefore invoke :func:`try_clause_fix` first; if it returns
    ``None`` (no ``clause_no_zero`` error or unfixable errors present) they may
    then try :func:`try_preamble_fix` directly.

    Args:
        text: Raw text to fix.  When *llm_response* is ``True`` the DIMACS
            content is expected inside ``<cnf>…</cnf>`` tags.
        llm_response: Whether *text* comes from an LLM response (with tags).

    Returns:
        A corrected DIMACS string, or ``None`` if ``clause_no_zero`` is not
        among the errors or if unfixable errors remain.
    """
    cnf, error_message, status = parse_dimacs(text, llm_response)
    if status != CorrectionStatus.INVALID:
        return None

    _, full_error = cnf.is_valid(error_message)
    codes = classify_error_types(full_error)
    # Accept any mix of clause-zero and preamble errors; reject anything else
    # (e.g. invalid_header, empty_clause) that cannot be auto-repaired.
    fixable = CLAUSE_ERROR_CODES | PREAMBLE_ERROR_CODES
    if not codes or "clause_no_zero" not in codes:
        return None
    if not all(c in fixable for c in codes):
        return None

    if llm_response:
        dimacs = _extract_cnf_block(text)
        if dimacs is None:
            return None
    else:
        dimacs = text.strip()

    fixed_lines = []
    for line in dimacs.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("c") or stripped.startswith("p"):
            fixed_lines.append(line)
            continue
        try:
            parts = list(map(int, stripped.split()))
            if parts and parts[-1] != 0:
                line = stripped + " 0"
        except ValueError:
            pass
        fixed_lines.append(line)

    fixed_dimacs = "\n".join(fixed_lines)
    candidate = f"<cnf>\n{fixed_dimacs}\n</cnf>" if llm_response else fixed_dimacs
    return try_preamble_fix(candidate, llm_response) or candidate
