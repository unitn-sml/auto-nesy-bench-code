"""Model counting (exact and ApproxMC) and confusion-matrix counts between CNFs."""

import io
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import threading
from typing import Iterable, Optional, Sequence

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.allies.approxmc import Counter

from fonesy.general_utils import log
from fonesy.evaluation.procutil import set_pdeathsig
from fonesy.utils.oracles import TARGETS_DIR as _TARGETS_DIR

# Seconds allowed for a single model-count call.
_DEFAULT_TIMEOUT = 600

# Formulas with at most this many variables are counted exactly by
# enumeration; larger ones with ApproxMC.
_EXACT_MAX_VARS = 20

# ApproxMC tolerance and confidence.
_DEFAULT_EPSILON = 0.1
_DEFAULT_DELTA = 0.05


# pyapproxmc 4.3.2 miscounts projected formulas (``x1 <-> x2`` projected on
# x1 yields 1 model instead of 2); requirements.txt pins 4.1.24. The check
# below runs once per process before the first projected approximate count.
_projection_sanity_lock = threading.Lock()
_projection_sanity_result: Optional[str] = None


def _worker(fn, args, result_queue) -> None:
    set_pdeathsig()
    try:
        result_queue.put(("ok", fn(*args)))
    except Exception as e:  # noqa: BLE001 - forwarded to the parent as a message
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def _run_hard_timeout(fn, args, timeout: int, label: str):
    """Run ``fn(*args)`` in a spawned subprocess and SIGKILL it on timeout.

    Native SAT/ApproxMC calls do not return control to the interpreter, so an
    in-process timeout (``signal.alarm``) would never fire. ``spawn`` is used
    because the parent process may hold a CUDA context.

    Raises:
        TimeoutError: if *fn* does not finish within *timeout* seconds.
        RuntimeError: if *fn* raised inside the subprocess, or the subprocess
            exited without reporting a result (e.g. OOM-killed).
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(fn, args, result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.kill()
        process.join()
        raise TimeoutError(f"{label} timed out after {timeout}s")

    try:
        status, payload = result_queue.get(timeout=5)
    except Empty:
        raise RuntimeError(
            f"{label} subprocess exited without a result "
            f"(exitcode={process.exitcode})"
        )

    if status == "error":
        raise RuntimeError(payload)
    return payload


def _declared_num_vars(dimacs_str: str) -> int:
    """Variable count declared in the ``p cnf N M`` header.

    PySAT recomputes ``.nv`` from the literals that occur in the clauses;
    declared but unused (free) variables must still be counted.
    """
    for line in dimacs_str.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            return int(line.split()[2])
        break
    return 0


def load_precomputed_negation(problem_id: str, targets_dir: Path = _TARGETS_DIR) -> Optional[str]:
    """Return ``<targets_dir>/<problem_id>_negated.dimacs`` if it exists, else ``None``.

    These files are produced by ``ground-truth-dimacs/negate.py`` and only
    avoid re-negating large targets; without them the target is negated on
    the fly.
    """
    if not problem_id or "/" in problem_id or ".." in problem_id:
        return None
    path = Path(targets_dir) / f"{problem_id}_negated.dimacs"
    return path.read_text() if path.is_file() else None


def _precomputed_negation_metadata(dimacs_str: str) -> dict:
    """Parse the ``c source_nv:`` / ``c source_clauses:`` comments of a
    precomputed negation, used to check that it matches the current target."""
    meta = {}
    for line in dimacs_str.splitlines():
        line = line.strip()
        if line.startswith("p"):
            break
        if line.startswith("c source_nv:"):
            meta["source_nv"] = int(line.split(":", 1)[1])
        elif line.startswith("c source_clauses:"):
            meta["source_clauses"] = int(line.split(":", 1)[1])
    return meta


def _variables_in_clauses(cnf: CNF) -> set[int]:
    """Return the positive variable IDs that syntactically occur in *cnf*."""
    return {abs(lit) for clause in cnf.clauses for lit in clause}


def _normalise_projection_vars(
    projection_vars: Optional[Iterable[int]], declared_num_vars: int
) -> Optional[tuple[int, ...]]:
    """Validate a public ``projection_vars`` argument and freeze its order."""
    if projection_vars is None:
        return None
    if isinstance(projection_vars, (str, bytes)):
        raise ValueError("projection_vars must be an iterable of positive integers")
    try:
        projection = tuple(projection_vars)
    except TypeError as exc:
        raise ValueError(
            "projection_vars must be an iterable of positive integers"
        ) from exc
    if any(
        isinstance(var, bool) or not isinstance(var, int) or var <= 0
        for var in projection
    ):
        raise ValueError("projection_vars must contain only positive integers")
    if len(set(projection)) != len(projection):
        raise ValueError("projection_vars must not contain duplicates")
    if projection and max(projection) > declared_num_vars:
        raise ValueError(
            "projection_vars contains a variable outside the DIMACS header "
            f"(max={max(projection)}, declared={declared_num_vars})"
        )
    return projection


def _ensure_approxmc_projection_sane() -> None:
    """Fail fast if the installed ApproxMC miscounts projected formulas."""
    global _projection_sanity_result
    with _projection_sanity_lock:
        if _projection_sanity_result is None:
            equivalence = CNF(from_clauses=[[-1, 2], [1, -2]])
            try:
                with Counter(
                    formula=equivalence,
                    epsilon=_DEFAULT_EPSILON,
                    delta=_DEFAULT_DELTA,
                ) as counter:
                    observed = counter.count(projection=[1])
                if observed == 2:
                    _projection_sanity_result = "ok"
                else:
                    _projection_sanity_result = (
                        "ApproxMC projected-count sanity check failed: "
                        f"x1 <-> x2 projected onto x1 should have 2 models, got {observed}. "
                        "Install the pinned pyapproxmc==4.1.24 dependency and rerun."
                    )
            except Exception as exc:  # noqa: BLE001 - actionable backend error
                _projection_sanity_result = (
                    "ApproxMC projected-count sanity check failed: "
                    f"{type(exc).__name__}: {exc}. Install the pinned "
                    "pyapproxmc==4.1.24 dependency and rerun."
                )
    if _projection_sanity_result != "ok":
        raise RuntimeError(_projection_sanity_result)


def _compute_model_count_approx(
    dimacs_str: str,
    epsilon: float,
    delta: float,
    nv: int,
    projection_vars: Optional[tuple[int, ...]],
) -> int:
    cnf = CNF(from_string=dimacs_str)
    used = _variables_in_clauses(cnf)
    if projection_vars is None:
        # ApproxMC counts every ID up to the largest one in the clauses;
        # declared variables above it are free and multiplied in below.
        present_projection = sorted(used)
        free_projection_count = max(0, nv - cnf.nv)
    else:
        # Projected variables absent from the clauses are free.
        present_projection = [var for var in projection_vars if var in used]
        free_projection_count = len(projection_vars) - len(present_projection)

    # An empty projection has one model iff the formula is satisfiable.
    if not present_projection:
        with Solver(name="g3", bootstrap_with=cnf.clauses) as solver:
            count = int(solver.solve())
    else:
        with Counter(formula=cnf, epsilon=epsilon, delta=delta) as counter:
            if projection_vars is None:
                count = counter.count()
            else:
                count = counter.count(projection=present_projection)

    if free_projection_count:
        count *= 2 ** free_projection_count
    return count


def _compute_model_count_exact(
    dimacs_str: str,
    nv: int,
    projection_vars: Optional[tuple[int, ...]],
) -> int:
    """Exact (possibly projected) model count via enumerate-and-block."""
    cnf = CNF(from_string=dimacs_str)
    used = _variables_in_clauses(cnf)
    universe = tuple(range(1, nv + 1)) if projection_vars is None else projection_vars
    present_projection = tuple(var for var in universe if var in used)
    free_projection_count = len(universe) - len(present_projection)

    count = 0
    with Solver(name="g3", bootstrap_with=cnf.clauses) as solver:
        if not present_projection:
            count = int(solver.solve())
        else:
            while solver.solve():
                model_by_var = {abs(lit): lit for lit in solver.get_model()}
                projected_model = [model_by_var[var] for var in present_projection]
                count += 1
                solver.add_clause([-lit for lit in projected_model])

    if free_projection_count:
        count *= 2 ** free_projection_count
    return count


def model_count(
    dimacs_str: str,
    epsilon: float = _DEFAULT_EPSILON,
    delta: float = _DEFAULT_DELTA,
    timeout: int = _DEFAULT_TIMEOUT,
    max_exact_vars: int = _EXACT_MAX_VARS,
    *,
    projection_vars: Optional[Sequence[int]] = None,
) -> tuple:
    """Return ``(count, metadata)`` for a DIMACS formula.

    Formulas over at most *max_exact_vars* variables (after projection) are
    counted exactly, the others with ApproxMC(*epsilon*, *delta*).
    ``projection_vars``, when given, is the variable universe: every other
    variable is existentially quantified.
    """
    nv = _declared_num_vars(dimacs_str)
    projection = _normalise_projection_vars(projection_vars, nv)
    semantic_nv = nv if projection is None else len(projection)
    if semantic_nv <= max_exact_vars:
        count = _run_hard_timeout(
            _compute_model_count_exact,
            (dimacs_str, nv, projection),
            timeout,
            "model_count(exact)",
        )
        metadata = {"approximate": False, "epsilon": None, "delta": None}
        if projection is not None:
            metadata.update({"projected": True, "projection_num_vars": len(projection)})
        return count, metadata

    if projection is not None:
        _ensure_approxmc_projection_sane()
    count = _run_hard_timeout(
        _compute_model_count_approx,
        (dimacs_str, epsilon, delta, nv, projection),
        timeout,
        "model_count(approx)",
    )
    metadata = {"approximate": True, "epsilon": epsilon, "delta": delta}
    if projection is not None:
        metadata.update({"projected": True, "projection_num_vars": len(projection)})
    return count, metadata


def _cnf_to_dimacs_str(cnf: CNF) -> str:
    buf = io.StringIO()
    cnf.to_fp(buf)
    return buf.getvalue()


def combine_projected_dimacs(
    formulas: Sequence[str], primary_num_vars: int
) -> str:
    """Conjoin formulas that share only the primary variables ``1..primary_num_vars``.

    Auxiliary variables (IDs above that range) are local to each formula and
    are renamed so that they do not clash.
    """
    if isinstance(primary_num_vars, bool) or not isinstance(primary_num_vars, int):
        raise ValueError("primary_num_vars must be a non-negative integer")
    if primary_num_vars < 0:
        raise ValueError("primary_num_vars must be a non-negative integer")

    clauses: list[list[int]] = []
    next_aux = primary_num_vars
    for dimacs_str in formulas:
        cnf = CNF(from_string=dimacs_str)
        auxiliary_ids = sorted(
            var for var in _variables_in_clauses(cnf) if var > primary_num_vars
        )
        auxiliary_map = {
            old_id: next_aux + offset
            for offset, old_id in enumerate(auxiliary_ids, start=1)
        }
        next_aux += len(auxiliary_ids)
        for clause in cnf.clauses:
            clauses.append(
                [
                    (-1 if lit < 0 else 1)
                    * auxiliary_map.get(abs(lit), abs(lit))
                    for lit in clause
                ]
            )

    combined = CNF(from_clauses=clauses)
    combined.nv = next_aux
    return _cnf_to_dimacs_str(combined)


def _has_referenced_auxiliary(dimacs_str: str, primary_num_vars: int) -> bool:
    cnf = CNF(from_string=dimacs_str)
    return any(var > primary_num_vars for var in _variables_in_clauses(cnf))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _count_interval(count: int, metadata: dict, universe_size: int) -> tuple[int, int]:
    """Conservative integer interval for one exact/ApproxMC point estimate."""
    if not metadata.get("approximate"):
        value = _clamp(count, 0, universe_size)
        return value, value
    epsilon = metadata.get("epsilon")
    if epsilon is None or epsilon < 0:
        return 0, universe_size
    lower = math.floor(count / (1.0 + epsilon))
    upper = math.ceil(count * (1.0 + epsilon))
    return _clamp(lower, 0, universe_size), _clamp(upper, 0, universe_size)


def _include_point(interval: tuple[int, int], point: int) -> tuple[int, int]:
    return min(interval[0], point), max(interval[1], point)


def _projected_confusion_from_formulas(
    target_formula: str,
    pred_formula: str,
    intersection_formula: str,
    *,
    primary_num_vars: int,
    universe_size: int,
    epsilon: float,
    delta: float,
    timeout: int,
    max_exact_vars: int,
) -> dict:
    """Confusion counts for formulas with auxiliary variables.

    Negating a formula with (existentially quantified) auxiliaries is not
    sound, so we count T = MC(target), P = MC(pred) and TP = MC(target & pred)
    on the primary variables and derive FP = P - TP, FN = T - TP and
    TN = 2^n - T - P + TP. Each count uses confidence delta/3 (union bound).
    """
    if universe_size < 0 or universe_size > 2 ** primary_num_vars:
        raise ValueError(
            "universe_size must be between 0 and the primary assignment universe"
        )
    projection = tuple(range(1, primary_num_vars + 1))
    component_delta = delta / 3.0

    def _mc(formula: str):
        return model_count(
            formula,
            epsilon=epsilon,
            delta=component_delta,
            timeout=timeout,
            max_exact_vars=max_exact_vars,
            projection_vars=projection,
        )

    raw_target, target_meta = _mc(target_formula)
    raw_pred, pred_meta = _mc(pred_formula)
    raw_tp, intersection_meta = _mc(intersection_formula)

    target_count = _clamp(raw_target, 0, universe_size)
    pred_count = _clamp(raw_pred, 0, universe_size)
    feasible_tp_low = max(0, target_count + pred_count - universe_size)
    feasible_tp_high = min(target_count, pred_count)
    tp = _clamp(raw_tp, feasible_tp_low, feasible_tp_high)
    fp = pred_count - tp
    fn = target_count - tp
    tn = universe_size - target_count - pred_count + tp
    xor = fp + fn

    approximate = any(
        meta.get("approximate")
        for meta in (target_meta, pred_meta, intersection_meta)
    )
    reconciled = (target_count, pred_count, tp) != (raw_target, raw_pred, raw_tp)

    target_interval = _count_interval(raw_target, target_meta, universe_size)
    pred_interval = _count_interval(raw_pred, pred_meta, universe_size)
    tp_interval = _count_interval(raw_tp, intersection_meta, universe_size)
    tp_interval = (
        max(tp_interval[0], 0, target_interval[0] + pred_interval[0] - universe_size),
        min(tp_interval[1], target_interval[1], pred_interval[1]),
    )
    interval_guarantee_failed = tp_interval[0] > tp_interval[1]
    if interval_guarantee_failed:
        # The three confidence intervals are mutually inconsistent.
        tp_interval = (0, universe_size)
    tp_interval = _include_point(tp_interval, tp)

    fp_interval = _include_point(
        (
            max(0, pred_interval[0] - tp_interval[1]),
            min(universe_size, pred_interval[1] - tp_interval[0]),
        ),
        fp,
    )
    fn_interval = _include_point(
        (
            max(0, target_interval[0] - tp_interval[1]),
            min(universe_size, target_interval[1] - tp_interval[0]),
        ),
        fn,
    )
    tn_interval = _include_point(
        (
            max(
                0,
                universe_size
                - target_interval[1]
                - pred_interval[1]
                + tp_interval[0],
            ),
            min(
                universe_size,
                universe_size
                - target_interval[0]
                - pred_interval[0]
                + tp_interval[1],
            ),
        ),
        tn,
    )
    xor_interval = _include_point(
        (
            max(
                0,
                target_interval[0] + pred_interval[0] - 2 * tp_interval[1],
            ),
            min(
                universe_size,
                target_interval[1] + pred_interval[1] - 2 * tp_interval[0],
            ),
        ),
        xor,
    )
    if interval_guarantee_failed:
        fp_interval = fn_interval = tn_interval = xor_interval = (
            0,
            universe_size,
        )

    source_counts = {
        "target": raw_target,
        "prediction": raw_pred,
        "intersection": raw_tp,
    }

    def _meta(interval: tuple[int, int], *, derived: bool) -> dict:
        return {
            "approximate": approximate,
            "epsilon": epsilon if approximate else None,
            # Union bound: each of T, P and T∩P uses delta/3.
            "delta": delta if approximate else None,
            "component_delta": component_delta if approximate else None,
            "projected": True,
            "primary_num_vars": primary_num_vars,
            "universe_size": universe_size,
            "derived": derived,
            "individual_relative_error_guarantee": not (approximate and derived),
            "reconciled": reconciled,
            "interval_guarantee_failed": interval_guarantee_failed,
            "interval": interval,
            "source_counts": source_counts,
        }

    tp_was_reconciled = tp != raw_tp or interval_guarantee_failed
    return {
        "tp": (tp, _meta(tp_interval, derived=tp_was_reconciled)),
        "fp": (fp, _meta(fp_interval, derived=True)),
        "fn": (fn, _meta(fn_interval, derived=True)),
        "tn": (tn, _meta(tn_interval, derived=True)),
        "xor": (xor, _meta(xor_interval, derived=True)),
        # An approximate zero is not a proof of equivalence.
        "equal": None if approximate else xor == 0,
    }


def _build_confusion_dimacs(
    target_str: str,
    pred_str: str,
    precomputed_negated_target: Optional[str] = None,
) -> dict:
    """Build the DIMACS of target&pred, ~target&pred, target&~pred and ~target&~pred.

    Negations use PySAT's Tseitin ``CNF.negate``. *precomputed_negated_target*
    (see :func:`load_precomputed_negation`) replaces ``target.negate()`` when it
    is identical to it, i.e. when the prediction does not declare more
    variables than the target.

    Returns ``{"tp": (dimacs_str, nv, n_clauses), "fp": ..., "fn": ..., "tn": ...}``.
    """
    target = CNF(from_string=target_str)
    target.nv = _declared_num_vars(target_str)
    pred = CNF(from_string=pred_str)
    pred.nv = _declared_num_vars(pred_str)
    base_nv = max(target.nv, pred.nv)

    # ~target and ~pred are never conjoined in the fp/fn formulas, so both
    # number their Tseitin variables from base_nv + 1 (a gap would add free
    # variables and inflate the counts).
    not_target = None
    if precomputed_negated_target is not None and pred.nv <= target.nv:
        meta = _precomputed_negation_metadata(precomputed_negated_target)
        if meta.get("source_nv") == target.nv and meta.get("source_clauses") == len(target.clauses):
            not_target = CNF(from_string=precomputed_negated_target)
            not_target.nv = _declared_num_vars(precomputed_negated_target)
        else:
            log(
                "Precomputed negated target metadata mismatch "
                f"(expected source_nv={target.nv}/source_clauses={len(target.clauses)}, "
                f"got {meta}) — falling back to on-the-fly negation",
                "WARNING",
            )
    if not_target is None:
        not_target = target.negate(topv=base_nv)

    not_pred = pred.negate(topv=base_nv)

    # ~target & ~pred: the two negations need disjoint Tseitin variables.
    not_pred_for_tn = pred.negate(topv=not_target.nv)

    def _combine(clauses_a, nv_a, clauses_b, nv_b) -> CNF:
        combined = CNF(from_clauses=clauses_a + clauses_b)
        combined.nv = max(nv_a, nv_b)
        return combined

    tp_cnf = _combine(target.clauses, target.nv, pred.clauses, pred.nv)
    fp_cnf = _combine(not_target.clauses, not_target.nv, pred.clauses, pred.nv)
    fn_cnf = _combine(target.clauses, target.nv, not_pred.clauses, not_pred.nv)
    tn_cnf = _combine(not_target.clauses, not_target.nv, not_pred_for_tn.clauses, not_pred_for_tn.nv)

    return {
        "tp": (_cnf_to_dimacs_str(tp_cnf), tp_cnf.nv, len(tp_cnf.clauses)),
        "fp": (_cnf_to_dimacs_str(fp_cnf), fp_cnf.nv, len(fp_cnf.clauses)),
        "fn": (_cnf_to_dimacs_str(fn_cnf), fn_cnf.nv, len(fn_cnf.clauses)),
        "tn": (_cnf_to_dimacs_str(tn_cnf), tn_cnf.nv, len(tn_cnf.clauses)),
    }


def confusion_counts(
    target_str: str,
    pred_str: str,
    epsilon: float = _DEFAULT_EPSILON,
    delta: float = _DEFAULT_DELTA,
    timeout: int = _DEFAULT_TIMEOUT,
    max_exact_vars: int = _EXACT_MAX_VARS,
    precomputed_negated_target: Optional[str] = None,
    *,
    primary_num_vars: Optional[int] = None,
) -> dict:
    """Confusion-matrix model counts between a target and a predicted CNF.

    Returns ``{"tp", "fp", "fn", "tn", "xor"}`` mapped to ``(count, metadata)``
    with TP = MC(target & pred), FP = MC(~target & pred),
    FN = MC(target & ~pred), TN = MC(~target & ~pred) and XOR = FP + FN, plus
    ``"equal"``: ``True``/``False`` when the counts are exact, ``None`` when
    approximate (an approximate zero does not prove equivalence).

    Without auxiliary variables all four cells are counted independently.
    When *primary_num_vars* is given and either formula uses variables above
    it (e.g. Tseitin variables), those are existentially quantified and the
    cells are derived from projected counts (see
    :func:`_projected_confusion_from_formulas`).
    """
    if primary_num_vars is not None:
        if (
            isinstance(primary_num_vars, bool)
            or not isinstance(primary_num_vars, int)
            or primary_num_vars < 0
        ):
            raise ValueError("primary_num_vars must be a non-negative integer")

        target_has_aux = _has_referenced_auxiliary(target_str, primary_num_vars)
        pred_has_aux = _has_referenced_auxiliary(pred_str, primary_num_vars)
        if target_has_aux or pred_has_aux:
            target_formula = combine_projected_dimacs([target_str], primary_num_vars)
            pred_formula = combine_projected_dimacs([pred_str], primary_num_vars)
            intersection_formula = combine_projected_dimacs(
                [target_str, pred_str], primary_num_vars
            )
            return _projected_confusion_from_formulas(
                target_formula,
                pred_formula,
                intersection_formula,
                primary_num_vars=primary_num_vars,
                universe_size=2 ** primary_num_vars,
                epsilon=epsilon,
                delta=delta,
                timeout=timeout,
                max_exact_vars=max_exact_vars,
            )

        # Normalize both headers to exactly the primary variables.
        target_str = combine_projected_dimacs([target_str], primary_num_vars)
        pred_str = combine_projected_dimacs([pred_str], primary_num_vars)

    formulas = _build_confusion_dimacs(
        target_str, pred_str, precomputed_negated_target=precomputed_negated_target
    )

    def _mc(key: str):
        dimacs_str, _nv, _n_clauses = formulas[key]
        return model_count(
            dimacs_str, epsilon=epsilon, delta=delta,
            timeout=timeout, max_exact_vars=max_exact_vars,
        )

    tp_count, tp_meta = _mc("tp")
    fp_count, fp_meta = _mc("fp")
    fn_count, fn_meta = _mc("fn")
    tn_count, tn_meta = _mc("tn")

    xor_approx = fp_meta["approximate"] or fn_meta["approximate"]
    xor_meta = {
        "approximate": xor_approx,
        "epsilon": epsilon if xor_approx else None,
        "delta": delta if xor_approx else None,
    }

    result = {
        "tp": (tp_count, tp_meta),
        "fp": (fp_count, fp_meta),
        "fn": (fn_count, fn_meta),
        "tn": (tn_count, tn_meta),
        "xor": (fp_count + fn_count, xor_meta),
    }
    result["equal"] = None if xor_approx else result["xor"][0] == 0
    return result
