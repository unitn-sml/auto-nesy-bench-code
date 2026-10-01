"""Model-counting based evaluation of predicted CNFs against the ground truth.

For every valid prediction ``pred`` and target ``target`` we count

    TP = MC(target & pred)     FP = MC(~target & pred)
    FN = MC(target & ~pred)    TN = MC(~target & ~pred)

(exactly for small formulas, with ApproxMC otherwise, see
:mod:`fonesy.evaluation.solver`) and derive precision, recall, F1 and accuracy.
Model counting can optionally run in a pool of background processes
(:class:`fonesy.evaluation.mc_pool.McWorkerPool`) so that it never blocks LLM
generation.
"""

import decimal
import time
from dataclasses import dataclass
from typing import Optional

from fonesy.evaluation.dimacs import (
    ERROR_TYPE_CODES,
    canonicalize_primary_variables,
    classify_error_types,
    verify_cnf as _dimacs_verify_cnf,
)
from fonesy.evaluation.formula import validate_primary_num_vars
from fonesy.evaluation.mc_pool import McWorkerPool
from fonesy.evaluation.solver import confusion_counts, load_precomputed_negation, model_count
from fonesy.general_utils import log
from fonesy.statistics.statistics import CorrectionStatus

_DEFAULT_HARVEST_TIMEOUT = 600


@dataclass
class SampleOutcome:
    """Everything recorded for one evaluated sample."""

    is_valid: bool
    error_message: Optional[str]
    syntax_error: bool
    fixable_syntax_error: bool
    tseitin_converted: bool
    equal: Optional[bool]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    accuracy: Optional[float]
    tp: Optional[int]
    fp: Optional[int]
    fn: Optional[int]
    tn: Optional[int]
    tp_meta: Optional[dict]
    fp_meta: Optional[dict]
    fn_meta: Optional[dict]
    tn_meta: Optional[dict]
    mc_target_xor_pred: Optional[int]
    mc_target_xor_pred_meta: Optional[dict]
    formula_length: dict
    num_clauses: dict
    num_vars: dict
    model_count: dict
    model_count_meta: dict


# Target model counts known in closed form (all complete 4x4 Sudoku grids).
_KNOWN_TARGET_MODEL_COUNTS = {"sudoku": 288}


def _safe_div(numerator, denominator) -> float:
    return numerator / denominator if denominator else 0.0


def _formula_length(cnf) -> int:
    return sum(len(c) for c in cnf.clauses)


# The two functions below may run inside a spawned pool worker, where the
# log handlers of the main process are not configured: they return warnings
# as data instead of logging them.


def _cnf_structural_stats(cnf, label: str, problem_id: str, cc_kwargs: dict) -> dict:
    """Length, clause count, variable count and model count of one formula."""
    if cnf is None:
        return {
            "formula_length": None, "num_clauses": None, "num_vars": None,
            "model_count": None, "model_count_meta": None, "warning": None,
        }
    if label == "target" and problem_id in _KNOWN_TARGET_MODEL_COUNTS:
        count = _KNOWN_TARGET_MODEL_COUNTS[problem_id]
        meta = {"approximate": False, "epsilon": None, "delta": None}
        warning = None
    else:
        try:
            kwargs = dict(cc_kwargs)
            # Auxiliary (e.g. Tseitin) variables are projected away.
            if getattr(cnf, "has_auxiliary_variables", False):
                kwargs["projection_vars"] = cnf.projection_vars
            count, meta = model_count(cnf.to_dimacs(), **kwargs)
            warning = None
        except Exception as e:
            warning = f"Model count failed for {label}: {e}"
            count, meta = None, None
    return {
        "formula_length": _formula_length(cnf),
        "num_clauses": len(cnf.clauses),
        "num_vars": cnf.num_vars,
        "model_count": count,
        "model_count_meta": meta,
        "warning": warning,
    }


def _pair_stats(pred_s: dict, tgt_s: dict) -> dict:
    return {
        key: {"pred": pred_s[key], "target": tgt_s[key]}
        for key in ("formula_length", "num_clauses", "num_vars", "model_count", "model_count_meta")
    }


def _compute_mc_dependent_metrics(target_cnf, pred_cnf, problem_id: str, cc_kwargs: dict) -> dict:
    """Structural statistics and confusion-matrix metrics for one valid prediction."""
    pred_s = _cnf_structural_stats(pred_cnf, "pred", problem_id, cc_kwargs)
    tgt_s = _cnf_structural_stats(target_cnf, "target", problem_id, cc_kwargs)
    warnings = [w for w in (pred_s["warning"], tgt_s["warning"]) if w]

    try:
        kwargs = {"precomputed_negated_target": load_precomputed_negation(problem_id), **cc_kwargs}
        # With auxiliary variables on either side, counting is projected on
        # the primary variables of the target.
        if getattr(target_cnf, "has_auxiliary_variables", False) or getattr(
            pred_cnf, "has_auxiliary_variables", False
        ):
            kwargs["primary_num_vars"] = target_cnf.primary_num_vars
        counts = confusion_counts(target_cnf.to_dimacs(), pred_cnf.to_dimacs(), **kwargs)
        tp, tp_meta = counts["tp"]
        fp, fp_meta = counts["fp"]
        fn, fn_meta = counts["fn"]
        tn, tn_meta = counts["tn"]
        mc_xor, mc_xor_meta = counts["xor"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
        # An approximate zero is not a proof of equivalence: the solver may
        # return an explicit tri-state (True / False / None = unknown).
        equal = counts.get("equal", mc_xor == 0)
    except Exception as e:
        warnings.append(f"Model counting failed: {e}")
        tp = fp = fn = tn = None
        tp_meta = fp_meta = fn_meta = tn_meta = None
        mc_xor = mc_xor_meta = None
        precision = recall = f1 = accuracy = None
        equal = None  # evaluator failure: unknown, not a semantic error

    return {
        "stats": _pair_stats(pred_s, tgt_s),
        "equal": equal, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tp_meta": tp_meta, "fp_meta": fp_meta, "fn_meta": fn_meta, "tn_meta": tn_meta,
        "mc_xor": mc_xor, "mc_xor_meta": mc_xor_meta,
        "warnings": warnings,
    }


def _null_mc_stats() -> dict:
    return {
        key: {"pred": None, "target": None}
        for key in ("formula_length", "num_clauses", "num_vars", "model_count", "model_count_meta")
    }


def _null_mc_result(warning: Optional[str]) -> dict:
    """Result of a background job that failed or timed out (equality unknown)."""
    return {
        "stats": _null_mc_stats(),
        "equal": None, "precision": None, "recall": None, "f1": None, "accuracy": None,
        "tp": None, "fp": None, "fn": None, "tn": None,
        "tp_meta": None, "fp_meta": None, "fn_meta": None, "tn_meta": None,
        "mc_xor": None, "mc_xor_meta": None,
        "warnings": [warning] if warning else [],
    }


@dataclass
class _PendingEntry:
    """A sample submitted to the background pool and not yet harvested."""

    job_id: int
    list_index: int
    sample_key: object
    syntax_error: bool
    tseitin_converted: bool


class Metrics:
    """Accumulate per-sample outcomes and aggregate them with :meth:`compute`.

    Args:
        error_type_codes: Error codes of the output format (defaults to DIMACS).
        classify_errors_fn: Maps an error message to its error codes.
        model_count_timeout, epsilon, delta, max_exact_vars: Optional overrides
            of the model-counting defaults of :mod:`fonesy.evaluation.solver`.
        mc_workers: Number of background model-counting processes used by
            :meth:`submit`; ``0`` makes :meth:`submit` synchronous.
    """

    # Significant digits of the Decimal fallback in _avg.
    _AVG_DECIMAL_PREC = 50

    def __init__(
        self,
        error_type_codes=None,
        classify_errors_fn=None,
        model_count_timeout=None,
        epsilon=None,
        delta=None,
        max_exact_vars=None,
        mc_workers: int = 0,
    ):
        self._mc_workers = mc_workers
        self._pool = None
        self._pending: list = []

        self._error_type_codes = error_type_codes if error_type_codes is not None else ERROR_TYPE_CODES
        self._classify_errors_fn = classify_errors_fn if classify_errors_fn is not None else classify_error_types
        self._model_count_timeout = model_count_timeout
        self._epsilon = epsilon
        self._delta = delta
        self._max_exact_vars = max_exact_vars

        self.total = 0
        self.syntax_errors = 0
        self.fixable_syntax_errors = 0  # syntax errors that were later fixed
        self.semantic_errors = 0
        self.equal_count = 0
        self.semantic_unknowns = 0  # equivalence could not be decided
        self.tseitin_conversions = 0

        # Per-sample values, aligned by index.
        self._formula_lengths = []
        self._num_clauses = []
        self._num_vars = []
        self._model_counts = []
        self._model_count_metas = []
        self._problem_ids = []
        self._precisions = []
        self._recalls = []
        self._f1s = []
        self._accuracies = []
        self._equals = []
        self._tps = []
        self._fps = []
        self._fns = []
        self._tns = []
        self._tp_metas = []
        self._fp_metas = []
        self._fn_metas = []
        self._tn_metas = []
        self._mc_xors = []
        self._mc_xor_metas = []

        self._error_type_counts: dict = {code: 0 for code in self._error_type_codes}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _model_count_kwargs(self) -> dict:
        """Only the explicitly overridden model-counting parameters."""
        kwargs = {}
        if self._model_count_timeout is not None:
            kwargs["timeout"] = self._model_count_timeout
        if self._epsilon is not None:
            kwargs["epsilon"] = self._epsilon
        if self._delta is not None:
            kwargs["delta"] = self._delta
        if self._max_exact_vars is not None:
            kwargs["max_exact_vars"] = self._max_exact_vars
        return kwargs

    def _record_error_types(self, error_message: str) -> None:
        for code in self._classify_errors_fn(error_message):
            if code in self._error_type_counts:
                self._error_type_counts[code] += 1

    @staticmethod
    def _avg(values):
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        total, n = sum(valid), len(valid)
        try:
            return total / n
        except OverflowError:
            # Model counts of nearly unconstrained formulas can exceed the
            # float range; average them exactly with Decimal instead.
            with decimal.localcontext() as ctx:
                ctx.prec = Metrics._AVG_DECIMAL_PREC
                return decimal.Decimal(total) / decimal.Decimal(n)

    @staticmethod
    def _avg_pair(pair_list, key):
        return Metrics._avg([p[key] for p in pair_list])

    @staticmethod
    def _frac_approx(meta_list):
        valid = [m for m in meta_list if m is not None]
        if not valid:
            return None
        return sum(1 for m in valid if m["approximate"]) / len(valid)

    def _append_sample_metrics(
        self, problem_id, equal, precision, recall, f1, accuracy, tp, fp, fn, tn,
        tp_meta, fp_meta, fn_meta, tn_meta, mc_xor, mc_xor_meta,
    ):
        self._problem_ids.append(problem_id)
        self._equals.append(equal)
        self._precisions.append(precision)
        self._recalls.append(recall)
        self._f1s.append(f1)
        self._accuracies.append(accuracy)
        self._tps.append(tp)
        self._fps.append(fp)
        self._fns.append(fn)
        self._tns.append(tn)
        self._tp_metas.append(tp_meta)
        self._fp_metas.append(fp_meta)
        self._fn_metas.append(fn_meta)
        self._tn_metas.append(tn_meta)
        self._mc_xors.append(mc_xor)
        self._mc_xor_metas.append(mc_xor_meta)

    def _patch_sample_metrics(self, index: int, outcome: SampleOutcome) -> None:
        """Fill a slot reserved by :meth:`_reserve_pending_slot`."""
        self._equals[index] = outcome.equal
        self._precisions[index] = outcome.precision
        self._recalls[index] = outcome.recall
        self._f1s[index] = outcome.f1
        self._accuracies[index] = outcome.accuracy
        self._tps[index] = outcome.tp
        self._fps[index] = outcome.fp
        self._fns[index] = outcome.fn
        self._tns[index] = outcome.tn
        self._tp_metas[index] = outcome.tp_meta
        self._fp_metas[index] = outcome.fp_meta
        self._fn_metas[index] = outcome.fn_meta
        self._tn_metas[index] = outcome.tn_meta
        self._mc_xors[index] = outcome.mc_target_xor_pred
        self._mc_xor_metas[index] = outcome.mc_target_xor_pred_meta

    # ------------------------------------------------------------------
    # Shared steps of update() and submit()
    # ------------------------------------------------------------------

    def _validate_target(self, target_text, primary_num_vars=None):
        primary_num_vars = validate_primary_num_vars(primary_num_vars)
        target_cnf, error_message, status = _dimacs_verify_cnf(target_text, False)
        if target_cnf is None or status == CorrectionStatus.INVALID:
            log(f"Invalid target CNF: {error_message}", "CRITICAL")
            raise ValueError("Invalid target DIMACS — check your dataset")
        if primary_num_vars is not None:
            if primary_num_vars > target_cnf.num_vars:
                raise ValueError(
                    f"primary_num_vars ({primary_num_vars}) exceeds target "
                    f"DIMACS num_vars ({target_cnf.num_vars})"
                )
            target_cnf = canonicalize_primary_variables(target_cnf, primary_num_vars)
        return target_cnf

    @staticmethod
    def _canonicalize_prediction(pred_cnf, primary_num_vars):
        """Move a valid prediction into the target's primary-variable universe,
        keeping non-structural attributes (e.g. ``tseitin_converted``)."""
        if pred_cnf is None or primary_num_vars is None:
            return pred_cnf
        canonical = canonicalize_primary_variables(pred_cnf, primary_num_vars)
        structural_attrs = {
            "num_vars", "num_clauses", "clauses", "raw",
            "primary_num_vars", "duplicate_header", "clause_before_header",
        }
        for name, value in vars(pred_cnf).items():
            if name not in structural_attrs:
                setattr(canonical, name, value)
        return canonical

    def _record_prediction_bookkeeping(self, pred_cnf, preamble_fix_error, initial_error) -> tuple:
        """Count Tseitin conversions and fixed syntax errors of a valid prediction.

        Returns ``(syntax_error, tseitin_converted)``.
        """
        tseitin_converted = getattr(pred_cnf, "tseitin_converted", False)
        if tseitin_converted:
            self.tseitin_conversions += 1

        original_error = preamble_fix_error if preamble_fix_error is not None else initial_error
        if original_error is None:
            return False, tseitin_converted
        self.syntax_errors += 1
        self.fixable_syntax_errors += 1
        self._record_error_types(original_error)
        return True, tseitin_converted

    def _finish_invalid_pred(self, error_message, target_cnf, problem_id: str) -> SampleOutcome:
        """An unparseable prediction is both a syntax and a semantic error."""
        log(f"Pred is invalid: {error_message}", "DEBUG")
        self.syntax_errors += 1
        self._record_error_types(error_message)
        self.semantic_errors += 1
        stats = self._record_structural_stats(pred_cnf=None, target_cnf=target_cnf, problem_id=problem_id)
        self._append_sample_metrics(
            problem_id, equal=False, precision=None, recall=None, f1=None, accuracy=None,
            tp=None, fp=None, fn=None, tn=None, tp_meta=None, fp_meta=None, fn_meta=None, tn_meta=None,
            mc_xor=None, mc_xor_meta=None,
        )
        return SampleOutcome(
            is_valid=False, error_message=error_message,
            syntax_error=True, fixable_syntax_error=False,
            tseitin_converted=False, equal=False,
            precision=None, recall=None, f1=None, accuracy=None,
            tp=None, fp=None, fn=None, tn=None, tp_meta=None, fp_meta=None, fn_meta=None, tn_meta=None,
            mc_target_xor_pred=None, mc_target_xor_pred_meta=None,
            **stats,
        )

    def _finalize_mc_result(self, result: dict, syntax_error: bool, tseitin_converted: bool) -> SampleOutcome:
        """Turn a model-counting result into a :class:`SampleOutcome` and
        update the equivalence counters."""
        for warning in result["warnings"]:
            log(warning, "WARNING")

        equal = result["equal"]
        if equal is True:
            self.equal_count += 1
        elif equal is False:
            self.semantic_errors += 1
        else:
            self.semantic_unknowns += 1

        return SampleOutcome(
            is_valid=True, error_message=None,
            syntax_error=syntax_error, fixable_syntax_error=syntax_error,
            tseitin_converted=tseitin_converted, equal=equal,
            precision=result["precision"], recall=result["recall"], f1=result["f1"], accuracy=result["accuracy"],
            tp=result["tp"], fp=result["fp"], fn=result["fn"], tn=result["tn"],
            tp_meta=result["tp_meta"], fp_meta=result["fp_meta"], fn_meta=result["fn_meta"], tn_meta=result["tn_meta"],
            mc_target_xor_pred=result["mc_xor"], mc_target_xor_pred_meta=result["mc_xor_meta"],
            **result["stats"],
        )

    def _finish_valid_pred_sync(
        self, target_cnf, pred_cnf, problem_id: str, syntax_error: bool, tseitin_converted: bool,
    ) -> SampleOutcome:
        result = _compute_mc_dependent_metrics(target_cnf, pred_cnf, problem_id, self._model_count_kwargs())
        self._append_structural_stats(result["stats"])
        outcome = self._finalize_mc_result(result, syntax_error, tseitin_converted)
        self._append_sample_metrics(
            problem_id, outcome.equal, outcome.precision, outcome.recall, outcome.f1, outcome.accuracy,
            outcome.tp, outcome.fp, outcome.fn, outcome.tn,
            outcome.tp_meta, outcome.fp_meta, outcome.fn_meta, outcome.tn_meta,
            outcome.mc_target_xor_pred, outcome.mc_target_xor_pred_meta,
        )
        return outcome

    def _prepare(self, pred_cnf, target_text, error_message, preamble_fix_error, initial_error,
                 problem_id, primary_num_vars):
        """Common prefix: returns ``(outcome, None)`` for invalid predictions,
        ``(None, (target_cnf, pred_cnf, syntax_error, tseitin_converted))`` otherwise."""
        self.total += 1
        target_cnf = self._validate_target(target_text, primary_num_vars)
        if pred_cnf is None:
            return self._finish_invalid_pred(
                error_message or "[ERROR]: Prediction did not produce a valid CNF\n",
                target_cnf,
                problem_id,
            ), None
        pred_cnf = self._canonicalize_prediction(pred_cnf, primary_num_vars)
        syntax_error, tseitin_converted = self._record_prediction_bookkeeping(
            pred_cnf, preamble_fix_error, initial_error
        )
        return None, (target_cnf, pred_cnf, syntax_error, tseitin_converted)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def update(
        self,
        pred_text,
        target_text,
        verify_cnf,
        preamble_fix_error=None,
        initial_error=None,
        problem_id: str = "",
        primary_num_vars=None,
    ):
        """Verify *pred_text* with *verify_cnf* and score it against *target_text*.

        Args:
            preamble_fix_error: Original error of a prediction repaired by the
                automatic fixer (mutually exclusive with *initial_error*).
            initial_error: Original error of a prediction repaired by the LLM.
            problem_id: Task id, used for per-task breakdowns and cached files.
            primary_num_vars: Number of primary variables; variables above it in
                either formula are treated as existentially quantified auxiliaries.
        """
        pred_cnf, error_message, status = verify_cnf(pred_text, True)
        return self.update_verified(
            pred_cnf,
            target_text,
            error_message=error_message if status == CorrectionStatus.INVALID else None,
            preamble_fix_error=preamble_fix_error,
            initial_error=initial_error,
            problem_id=problem_id,
            primary_num_vars=primary_num_vars,
        )

    def update_verified(
        self,
        pred_cnf,
        target_text,
        error_message=None,
        preamble_fix_error=None,
        initial_error=None,
        problem_id: str = "",
        primary_num_vars=None,
    ):
        """Like :meth:`update` for an already verified prediction (``None`` if invalid)."""
        outcome, prepared = self._prepare(
            pred_cnf, target_text, error_message, preamble_fix_error, initial_error,
            problem_id, primary_num_vars,
        )
        if outcome is not None:
            return outcome
        target_cnf, pred_cnf, syntax_error, tseitin_converted = prepared
        return self._finish_valid_pred_sync(target_cnf, pred_cnf, problem_id, syntax_error, tseitin_converted)

    # ------------------------------------------------------------------
    # Background API
    # ------------------------------------------------------------------

    def submit(
        self,
        pred_text,
        target_text,
        verify_cnf,
        preamble_fix_error=None,
        initial_error=None,
        problem_id: str = "",
        sample_key=None,
        verified_cnf=None,
        verification_error=None,
        already_verified: bool = False,
        primary_num_vars=None,
    ):
        """Background counterpart of :meth:`update`.

        Returns the :class:`SampleOutcome` right away when it can be computed
        synchronously (invalid prediction or ``mc_workers == 0``). Otherwise
        the model counting is submitted to the pool and ``None`` is returned;
        the outcome is later returned by :meth:`harvest_ready` / :meth:`harvest`
        as ``(sample_key, outcome)``. Aggregates keep submission order.
        """
        if already_verified:
            pred_cnf, error_message = verified_cnf, verification_error
        else:
            pred_cnf, error_message, status = verify_cnf(pred_text, True)
            error_message = error_message if status == CorrectionStatus.INVALID else None
        return self.submit_verified(
            pred_cnf,
            target_text,
            error_message=error_message,
            preamble_fix_error=preamble_fix_error,
            initial_error=initial_error,
            problem_id=problem_id,
            sample_key=sample_key,
            primary_num_vars=primary_num_vars,
        )

    def submit_verified(
        self,
        pred_cnf,
        target_text,
        error_message=None,
        preamble_fix_error=None,
        initial_error=None,
        problem_id: str = "",
        sample_key=None,
        primary_num_vars=None,
    ):
        """Like :meth:`submit` for an already verified prediction."""
        outcome, prepared = self._prepare(
            pred_cnf, target_text, error_message, preamble_fix_error, initial_error,
            problem_id, primary_num_vars,
        )
        if outcome is not None:
            return outcome
        target_cnf, pred_cnf, syntax_error, tseitin_converted = prepared

        if self._mc_workers <= 0:
            return self._finish_valid_pred_sync(target_cnf, pred_cnf, problem_id, syntax_error, tseitin_converted)
        if sample_key is None:
            raise ValueError("sample_key is required when submitting to the background pool (mc_workers > 0)")

        list_index = self._reserve_pending_slot(problem_id)
        job_id = self._get_or_create_pool().submit(
            _compute_mc_dependent_metrics, (target_cnf, pred_cnf, problem_id, self._model_count_kwargs())
        )
        self._pending.append(_PendingEntry(
            job_id=job_id, list_index=list_index, sample_key=sample_key,
            syntax_error=syntax_error, tseitin_converted=tseitin_converted,
        ))
        return None

    def _reserve_pending_slot(self, problem_id: str) -> int:
        """Append placeholders so per-sample lists follow submission order."""
        index = len(self._problem_ids)
        self._append_structural_stats(_null_mc_stats())
        self._append_sample_metrics(
            problem_id, equal=False, precision=None, recall=None, f1=None, accuracy=None,
            tp=None, fp=None, fn=None, tn=None, tp_meta=None, fp_meta=None, fn_meta=None, tn_meta=None,
            mc_xor=None, mc_xor_meta=None,
        )
        return index

    def _get_or_create_pool(self) -> McWorkerPool:
        if self._pool is None:
            self._pool = McWorkerPool(self._mc_workers)
        return self._pool

    def _finalize_pending(self, entry: _PendingEntry, status: str, payload) -> SampleOutcome:
        if status == "ok":
            result = payload
        elif status == "timeout":
            result = _null_mc_result(f"Background model counting timed out (job {entry.job_id})")
        else:
            result = _null_mc_result(f"Background model counting failed: {payload}")
        self._patch_structural_stats(entry.list_index, result["stats"])
        outcome = self._finalize_mc_result(result, entry.syntax_error, entry.tseitin_converted)
        self._patch_sample_metrics(entry.list_index, outcome)
        return outcome

    def _drain(self, deadline: float) -> list:
        if self._pool is None or not self._pending:
            return []
        results = self._pool.collect(deadline, n_expected=len(self._pending))
        resolved, still_pending = [], []
        for entry in self._pending:
            if entry.job_id in results:
                status, payload = results[entry.job_id]
                resolved.append((entry.sample_key, self._finalize_pending(entry, status, payload)))
            else:
                still_pending.append(entry)
        self._pending = still_pending
        return resolved

    def harvest_ready(self) -> list:
        """Collect the background jobs that already finished (non-blocking).

        Returns ``[(sample_key, SampleOutcome), ...]``.
        """
        return self._drain(deadline=time.monotonic())

    def harvest(self, timeout: int = _DEFAULT_HARVEST_TIMEOUT) -> list:
        """Wait up to *timeout* seconds in total for the outstanding jobs.

        Jobs still running at the deadline are recorded as failed model counts
        (equality unknown). The pool is shut down afterwards.
        Returns ``[(sample_key, SampleOutcome), ...]``.
        """
        resolved = self._drain(deadline=time.monotonic() + timeout)
        for entry in self._pending:
            resolved.append((entry.sample_key, self._finalize_pending(entry, "timeout", None)))
        self._pending = []
        self.shutdown()
        return resolved

    def shutdown(self) -> None:
        """Terminate the background pool (idempotent)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    # ------------------------------------------------------------------
    # Structural statistics
    # ------------------------------------------------------------------

    def _append_structural_stats(self, stats: dict) -> None:
        self._formula_lengths.append(stats["formula_length"])
        self._num_clauses.append(stats["num_clauses"])
        self._num_vars.append(stats["num_vars"])
        self._model_counts.append(stats["model_count"])
        self._model_count_metas.append(stats["model_count_meta"])

    def _patch_structural_stats(self, index: int, stats: dict) -> None:
        self._formula_lengths[index] = stats["formula_length"]
        self._num_clauses[index] = stats["num_clauses"]
        self._num_vars[index] = stats["num_vars"]
        self._model_counts[index] = stats["model_count"]
        self._model_count_metas[index] = stats["model_count_meta"]

    def _record_structural_stats(self, pred_cnf, target_cnf, problem_id: str = "") -> dict:
        cc_kwargs = self._model_count_kwargs()
        pred_s = _cnf_structural_stats(pred_cnf, "pred", problem_id, cc_kwargs)
        tgt_s = _cnf_structural_stats(target_cnf, "target", problem_id, cc_kwargs)
        for s in (pred_s, tgt_s):
            if s["warning"]:
                log(s["warning"], "WARNING")
        stats = _pair_stats(pred_s, tgt_s)
        self._append_structural_stats(stats)
        return stats

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute(self):
        """Aggregate metrics; averages skip samples where a value is undefined."""
        total = self.total
        return {
            "total": total,
            "syntax_error_rate": self.syntax_errors / total if total else 0,
            "fixable_syntax_rate": (
                self.fixable_syntax_errors / self.syntax_errors if self.syntax_errors else 0
            ),
            "semantic_error_rate": self.semantic_errors / total if total else 0,
            "semantic_unknown_count": self.semantic_unknowns,
            "semantic_unknown_rate": self.semantic_unknowns / total if total else 0,
            "equal_rate": self.equal_count / total if total else 0,
            "avg_precision": self._avg(self._precisions),
            "avg_recall": self._avg(self._recalls),
            "avg_f1": self._avg(self._f1s),
            "avg_accuracy": self._avg(self._accuracies),
            "avg_mc_target_xor_pred": self._avg(self._mc_xors),
            "frac_approx_tp": self._frac_approx(self._tp_metas),
            "frac_approx_fp": self._frac_approx(self._fp_metas),
            "frac_approx_fn": self._frac_approx(self._fn_metas),
            "frac_approx_tn": self._frac_approx(self._tn_metas),
            "frac_approx_xor": self._frac_approx(self._mc_xor_metas),
            "avg_formula_length": {
                "pred": self._avg_pair(self._formula_lengths, "pred"),
                "target": self._avg_pair(self._formula_lengths, "target"),
            },
            "avg_num_clauses": {
                "pred": self._avg_pair(self._num_clauses, "pred"),
                "target": self._avg_pair(self._num_clauses, "target"),
            },
            "avg_num_vars": {
                "pred": self._avg_pair(self._num_vars, "pred"),
                "target": self._avg_pair(self._num_vars, "target"),
            },
            "avg_model_count": {
                "pred": self._avg_pair(self._model_counts, "pred"),
                "target": self._avg_pair(self._model_counts, "target"),
            },
            "frac_approx_model_count_pred": self._frac_approx(
                [m["pred"] for m in self._model_count_metas]
            ),
            "frac_approx_model_count_target": self._frac_approx(
                [m["target"] for m in self._model_count_metas]
            ),
            "tseitin_conversion_count": self.tseitin_conversions,
            "tseitin_conversion_rate": self.tseitin_conversions / total if total else 0,
            "per_sample": {
                "formula_lengths": self._formula_lengths,
                "num_clauses": self._num_clauses,
                "num_vars": self._num_vars,
                "model_counts": self._model_counts,
                "precision": self._precisions,
                "recall": self._recalls,
                "f1": self._f1s,
                "accuracy": self._accuracies,
                "equal": self._equals,
                "tp": self._tps,
                "fp": self._fps,
                "fn": self._fns,
                "tn": self._tns,
                "mc_target_xor_pred": self._mc_xors,
            },
            "per_task_precision": dict(zip(self._problem_ids, self._precisions)),
            "per_task_recall": dict(zip(self._problem_ids, self._recalls)),
            "per_task_f1": dict(zip(self._problem_ids, self._f1s)),
            "per_task_accuracy": dict(zip(self._problem_ids, self._accuracies)),
            "error_type_counts": dict(self._error_type_counts),
        }
