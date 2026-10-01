"""Run-level statistics and correction-status tracking."""

from enum import Enum, auto
from typing import Dict


class CorrectionStatus(Enum):
    """Status of a CNF prediction or self-correction response."""

    OK = auto()  #: CNF was correct (no correction needed / model said OK).
    FIXED = auto()  #: CNF was corrected by the model.
    AUTO_FIXED = auto()  #: CNF preamble was corrected programmatically (no LLM query).
    INVALID = auto()  #: CNF was invalid and could not be fixed.


class Statistics:
    """Accumulate per-run counts for corrections and validation outcomes.

    Attributes:
        total: Total number of samples processed.
        correct_before: Samples already correct before self-correction.
        correct_after: Samples correct after self-correction.
        corrections_used: Samples where at least one LLM correction was applied.
        preamble_auto_corrected: Samples whose preamble was fixed programmatically.
        ok_from_model: Correction responses where the model said ``[[OK]]``.
        fixed_from_model: Correction responses where the model said ``[[FIXED]]``.
        invalid_from_model: Correction responses with unexpected output.
    """

    def __init__(self) -> None:
        self.total: int = 0
        self.correct_before: int = 0
        self.correct_after: int = 0

        self.ok_from_model: int = 0
        self.fixed_from_model: int = 0
        self.invalid_from_model: int = 0

        self.corrections_used: int = 0
        self.preamble_auto_corrected: int = 0

    def update(
        self,
        status_before: CorrectionStatus,
        status_after: CorrectionStatus,
        correction_used: CorrectionStatus,
        status_response: CorrectionStatus,
    ) -> None:
        """Record statistics for one sample.

        Args:
            status_before: Validation status before self-correction.
            status_after: Validation status after self-correction.
            correction_used: Whether a correction was applied
                (``FIXED``) or not (``INVALID``).
            status_response: The model's last correction response status.
        """
        self.total += 1

        if status_before == CorrectionStatus.OK:
            self.correct_before += 1

        if status_after == CorrectionStatus.OK:
            self.correct_after += 1

        if correction_used == CorrectionStatus.FIXED:
            self.corrections_used += 1

        if correction_used == CorrectionStatus.AUTO_FIXED:
            self.preamble_auto_corrected += 1

        if status_response == CorrectionStatus.OK:
            self.ok_from_model += 1
        elif status_response == CorrectionStatus.FIXED:
            self.fixed_from_model += 1
        else:
            self.invalid_from_model += 1

    def summary(self) -> Dict[str, int]:
        """Return a flat dictionary of all counters.

        Returns:
            Dictionary with keys: ``total``, ``correct_before``,
            ``correct_after``, ``corrections_used``, ``ok_from_model``,
            ``fixed_from_model``, ``invalid_from_model``.
        """
        return {
            "total": self.total,
            "correct_before": self.correct_before,
            "correct_after": self.correct_after,
            "corrections_used": self.corrections_used,
            "preamble_auto_corrected": self.preamble_auto_corrected,
            "ok_from_model": self.ok_from_model,
            "fixed_from_model": self.fixed_from_model,
            "invalid_from_model": self.invalid_from_model,
        }
