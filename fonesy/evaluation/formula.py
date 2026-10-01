"""DIMACS CNF object representation and variable-universe helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Tuple

from fonesy.statistics.statistics import CorrectionStatus


def validate_primary_num_vars(primary_num_vars: Optional[int]) -> Optional[int]:
    """Validate and return an expected number of primary variables.

    ``None`` means that no external primary-variable universe was supplied.
    This is configuration metadata rather than model output, so malformed
    values raise :class:`ValueError` instead of producing an ``INVALID``
    prediction.
    """
    if primary_num_vars is None:
        return None
    if isinstance(primary_num_vars, bool) or not isinstance(primary_num_vars, int):
        raise ValueError("primary_num_vars must be a non-negative integer")
    if primary_num_vars < 0:
        raise ValueError("primary_num_vars must be a non-negative integer")
    return primary_num_vars


def normalize_variable_names(
    variable_names: Optional[Sequence[str]],
) -> Optional[Tuple[str, ...]]:
    """Validate symbolic variable metadata and return an immutable tuple.

    The position in this tuple is authoritative: ``names[i]`` maps to DIMACS
    variable ``i + 1``.  The metadata belongs to the dataset/evaluation
    configuration, hence invalid metadata raises :class:`ValueError`.
    """
    if variable_names is None:
        return None
    if isinstance(variable_names, (str, bytes)):
        raise ValueError("variable_names must be a non-string sequence of names")
    if not isinstance(variable_names, Sequence):
        raise ValueError("variable_names must be a sequence of names")
    names = tuple(variable_names)
    if not names:
        raise ValueError("variable_names must contain at least one name")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("variable_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("variable_names must contain unique names")
    return names


class CNFFormula:
    """
    Object representation of a CNF formula in DIMACS format.
    """

    def __init__(
        self,
        num_vars: int,
        num_clauses: int,
        clauses: List[List[int]],
        raw: str,
        primary_num_vars: Optional[int] = None,
        duplicate_header: bool = False,
        clause_before_header: bool = False,
    ):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses
        self.raw = raw
        # Header problems detected while parsing, judged by `is_valid`.
        self.duplicate_header = duplicate_header
        self.clause_before_header = clause_before_header
        primary_num_vars = validate_primary_num_vars(primary_num_vars)
        # Without further context every declared variable is primary.
        self.primary_num_vars = (
            num_vars if primary_num_vars is None else primary_num_vars
        )

    @property
    def auxiliary_num_vars(self) -> int:
        """Number of physical variables outside the primary universe."""
        if self.num_vars is None or self.primary_num_vars is None:
            return 0
        return max(0, self.num_vars - self.primary_num_vars)

    @property
    def projection_vars(self) -> Tuple[int, ...]:
        """DIMACS variables onto which semantic evaluation must project."""
        if self.primary_num_vars is None:
            return ()
        return tuple(range(1, self.primary_num_vars + 1))

    @property
    def has_auxiliary_variables(self) -> bool:
        return self.auxiliary_num_vars > 0

    def is_valid(self, error_message) -> Tuple[CorrectionStatus, str]:
        preamble = "[ERROR]: "
        status = CorrectionStatus.OK

        if self.num_vars is None:
            error_message += preamble + "Missing number of variables\n"
            status = CorrectionStatus.INVALID

        if self.num_vars is not None and self.num_vars < 0:
            error_message += preamble + f"Negative declared num_vars: {self.num_vars}\n"
            status = CorrectionStatus.INVALID

        if self.num_clauses is not None and self.num_clauses < 0:
            error_message += preamble + f"Negative declared num_clauses: {self.num_clauses}\n"
            status = CorrectionStatus.INVALID

        if self.duplicate_header:
            error_message += preamble + "Duplicate header\n"
            status = CorrectionStatus.INVALID

        if self.clause_before_header:
            error_message += preamble + "Clause before header\n"
            status = CorrectionStatus.INVALID

        if self.num_clauses != len(self.clauses):
            error_message += preamble + (
                f"Clause count mismatch: expected {self.num_clauses}, "
                f"got {len(self.clauses)}\n"
            )
            status = CorrectionStatus.INVALID

        for i, clause in enumerate(self.clauses):
            if len(clause) == 0:
                error_message += preamble + f"Empty clause at index {i}\n"
                status = CorrectionStatus.INVALID

            if 0 in clause:
                error_message += preamble + f"Clause contains literal 0 (clause {i})\n"
                status = CorrectionStatus.INVALID

            if self.num_vars is not None:
                for lit in clause:
                    if abs(lit) > self.num_vars:
                        error_message += preamble + (
                            f"Literal {lit} exceeds num_vars={self.num_vars} (clause {i})\n"
                        )
                        status = CorrectionStatus.INVALID

        return status, error_message

    def clause_lengths(self):
        return [len(c) for c in self.clauses]

    def max_var_used(self):
        all_lits = [abs(lit) for c in self.clauses for lit in c]
        return max(all_lits) if all_lits else 0

    def to_dimacs(self) -> str:
        """
        Returns DIMACS string (safe for external solvers).
        """
        return self.raw

    def __repr__(self):
        return (
            f"CNFFormula(vars={self.num_vars}, primary={self.primary_num_vars}, "
            f"clauses={self.num_clauses})"
        )
