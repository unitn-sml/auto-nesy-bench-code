"""Metrics of the SPL / SL experiments: classification metrics, consistency
with the ground-truth formula, and the relation between two formulas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fonesy.spl_utilities.spl_problems import ProblemSpec


# ---------------------------------------------------------------------------
# Consistency with the ground-truth formula
# ---------------------------------------------------------------------------
# A MAP prediction always satisfies the formula the circuit was compiled from,
# so consistency is only informative for LLM-generated formulas: it counts the
# predictions that violate the ground-truth formula.


def build_ground_truth_masks(
    dimacs_path: str | Path, n_vars: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clause/literal masks ``(pos_mask, neg_mask)`` of the ground-truth CNF.

    ``pos_mask[c, v]`` (``neg_mask[c, v]``) is 1 iff variable ``v + 1`` occurs
    positively (negatively) in clause ``c``, so checking a batch against all
    clauses is two matrix products (targets have up to ~130k clauses). Circuit
    variables beyond the ground-truth ones are auxiliary and ignored.

    Raises ``ValueError`` if the CNF cannot be parsed or has more variables than
    the circuit.
    """
    from fonesy.evaluation.dimacs import verify_cnf

    dimacs_path = Path(dimacs_path)
    formula, err, _ = verify_cnf(dimacs_path.read_text())
    if formula is None:
        raise ValueError(f"Could not parse ground-truth CNF at {dimacs_path}: {err}")
    if formula.num_vars > n_vars:
        raise ValueError(
            f"Ground-truth CNF at {dimacs_path} declares {formula.num_vars} variables, "
            f"but the circuit only has {n_vars} — cannot check consistency."
        )

    n_oracle = formula.num_vars
    n_clauses = len(formula.clauses)
    pos_mask = torch.zeros(n_clauses, n_oracle, dtype=torch.float32)
    neg_mask = torch.zeros(n_clauses, n_oracle, dtype=torch.float32)
    for c, clause in enumerate(formula.clauses):
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_mask[c, v] = 1.0
            else:
                neg_mask[c, v] = 1.0
    return pos_mask.to(device), neg_mask.to(device)


# Larger formulas (kand-logic) are not compiled for the relation check.
_MAX_CLAUSES_EQCHECK = 20_000


def formula_vs_oracle(
    pred_dimacs: str | Path,
    oracle_dimacs: str | Path,
    auxiliary_num_vars: int = 0,
    max_clauses: int = _MAX_CLAUSES_EQCHECK,
) -> dict:
    """Relation (REL) and model-count ratio (MC-R) of a predicted CNF w.r.t. the ground truth.

    Both formulas are compiled to SDDs; the trailing ``auxiliary_num_vars``
    variables of the prediction P are existentially quantified out. With O the
    ground truth::

        O => P and P => O   -> "equal"
        O => P only         -> "permissive"    (admits invalid assignments)
        P => O only         -> "too_strict"    (excludes valid assignments)
        neither             -> "incomparable"

    Returns ``{"formula_rel": ..., "mc_ratio": MC(P) / MC(O)}``, with
    ``"unknown"`` / ``None`` if a formula cannot be parsed or compiled.
    """
    from fonesy.evaluation.dimacs import verify_cnf

    try:
        from pysdd.sdd import SddManager
    except ImportError:
        return {"formula_rel": "unknown", "mc_ratio": None}

    try:
        pf, _, _ = verify_cnf(Path(pred_dimacs).read_text())
        of, _, _ = verify_cnf(Path(oracle_dimacs).read_text())
    except OSError:
        return {"formula_rel": "unknown", "mc_ratio": None}
    if pf is None or of is None:
        return {"formula_rel": "unknown", "mc_ratio": None}
    if len(pf.clauses) > max_clauses or len(of.clauses) > max_clauses:
        return {"formula_rel": "unknown", "mc_ratio": None}

    n_pred = pf.num_vars
    primary = n_pred - (auxiliary_num_vars or 0)
    n = max(n_pred, of.num_vars)
    try:
        mgr = SddManager(var_count=n, auto_gc_and_minimize=True)

        def _build(clauses):
            node = mgr.true()
            for clause in clauses:
                disj = mgr.false()
                for lit in clause:
                    disj = disj | mgr.literal(lit)
                node = node & disj
            return node

        P = _build(pf.clauses)
        O = _build(of.clauses)
        if 0 < primary < n_pred:
            for v in range(primary + 1, n_pred + 1):
                P = mgr.exists(v, P)

        oracle_subset_pred = (O & ~P).is_false()
        pred_subset_oracle = (P & ~O).is_false()
        if oracle_subset_pred and pred_subset_oracle:
            rel = "equal"
        elif oracle_subset_pred:
            rel = "permissive"
        elif pred_subset_oracle:
            rel = "too_strict"
        else:
            rel = "incomparable"
        try:
            ratio = P.global_model_count() / max(O.global_model_count(), 1)
        except Exception:
            ratio = None
        return {"formula_rel": rel, "mc_ratio": ratio}
    except Exception:
        return {"formula_rel": "unknown", "mc_ratio": None}


def is_consistent(
    assignment: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor
) -> torch.Tensor:
    """``(B,)`` bool: whether each row of *assignment* satisfies every clause.

    Columns beyond the ground-truth variables (auxiliaries) are ignored.
    """
    n_oracle = pos_mask.shape[1]
    a = assignment[:, :n_oracle].float()   # drop auxiliary variables if present
    clause_has_true_pos_lit = (a @ pos_mask.T) > 0  # (B, n_clauses)
    clause_has_true_neg_lit = ((1.0 - a) @ neg_mask.T) > 0
    clause_satisfied = clause_has_true_pos_lit | clause_has_true_neg_lit
    return clause_satisfied.all(dim=1)


# ---------------------------------------------------------------------------
# Marginal probability extraction (needs a compiled circuit)
# ---------------------------------------------------------------------------


def make_integrate_query(circuit):
    """IntegrateQuery for *circuit*."""
    from cirkit.backend.torch.queries import IntegrateQuery
    return IntegrateQuery(circuit)


def _batch_size_from_kwargs(gate_function_kwargs: dict) -> int:
    return next(iter(gate_function_kwargs.values()))["x"].shape[0]


def _multi_label_marginal_probs(
    integrate_query, gate_function_kwargs: dict, spec: ProblemSpec, device
) -> torch.Tensor:
    """``(B, n_vars - eval_start_var)`` marginal probabilities P(bit = 1)."""
    batch_size = _batch_size_from_kwargs(gate_function_kwargs)
    n_vars = spec.n_vars
    eval_width = n_vars - spec.eval_start_var
    probs = torch.zeros(batch_size, eval_width, device=device)
    for j in range(eval_width):
        var_idx = spec.eval_start_var + j
        x = torch.zeros(batch_size, n_vars, device=device)
        x[:, var_idx] = 1.0
        integrate_vars = torch.ones(batch_size, n_vars, dtype=torch.bool, device=device)
        integrate_vars[:, var_idx] = False
        log_p = integrate_query(
            x,
            integrate_vars=integrate_vars,
            gate_function_kwargs=gate_function_kwargs,
        ).reshape(batch_size, -1)[:, 0]
        probs[:, j] = log_p.exp().clamp(0.0, 1.0)
    return probs


def _multiclass_marginal_probs(
    integrate_query, gate_function_kwargs: dict, spec: ProblemSpec, device
) -> torch.Tensor:
    """``(B, n_classes)`` class probabilities, normalized over the classes."""
    batch_size = _batch_size_from_kwargs(gate_function_kwargs)
    n_vars = spec.n_vars
    concept_end = spec.n_concept_vars
    integrate_vars_row = torch.ones(n_vars, dtype=torch.bool, device=device)
    integrate_vars_row[concept_end:] = False  # output bits are evidence, concepts marginalized
    integrate_vars = integrate_vars_row.unsqueeze(0).expand(batch_size, -1)

    probs = torch.zeros(batch_size, spec.n_classes, device=device)
    for k in range(spec.n_classes):
        bits = spec.class_to_output_bits(k).to(device)
        x = torch.zeros(batch_size, n_vars, device=device)
        x[:, concept_end:] = bits.unsqueeze(0).expand(batch_size, -1)
        log_p = integrate_query(
            x,
            integrate_vars=integrate_vars,
            gate_function_kwargs=gate_function_kwargs,
        ).reshape(batch_size, -1)[:, 0]
        probs[:, k] = log_p.exp().clamp(0.0, 1.0)

    row_sums = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs / row_sums


def compute_marginal_probs(
    integrate_query, gate_function_kwargs: dict, spec: ProblemSpec, device
) -> torch.Tensor:
    """Scores for AUROC/AUPRC: per-bit (multi-label) or per-class probabilities."""
    if spec.output_kind == "multi_label":
        return _multi_label_marginal_probs(integrate_query, gate_function_kwargs, spec, device)
    return _multiclass_marginal_probs(integrate_query, gate_function_kwargs, spec, device)


# ---------------------------------------------------------------------------
# Metric assembly (pure numpy/sklearn — independently unit-testable)
# ---------------------------------------------------------------------------


def _macro_binary_auc(y_true_indicator: np.ndarray, y_score: np.ndarray, metric_fn) -> float:
    """Macro-average a binary metric over the columns containing both classes (NaN if none)."""
    scores = []
    for j in range(y_true_indicator.shape[1]):
        col_true = y_true_indicator[:, j]
        if col_true.min() == col_true.max():
            continue
        scores.append(metric_fn(col_true, y_score[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def compute_classification_metrics(y_true, y_pred, y_score, spec: ProblemSpec) -> dict:
    """Macro accuracy, precision, recall, F1, AUROC and AUPRC.

    *y_true* / *y_pred* are ``(N, n_outputs)`` 0/1 arrays (multi-label) or
    ``(N,)`` class indices; *y_score* comes from :func:`compute_marginal_probs`.
    AUROC/AUPRC fall back to the accuracy when undefined.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)

    accuracy = float((y_true_arr == y_pred_arr).mean())
    precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    if spec.output_kind == "multi_label":
        y_true_indicator = y_true_arr
    else:
        y_true_indicator = np.eye(spec.n_classes, dtype=int)[y_true_arr]

    auroc = _macro_binary_auc(y_true_indicator, y_score_arr, roc_auc_score)
    auprc = _macro_binary_auc(y_true_indicator, y_score_arr, average_precision_score)
    if auroc != auroc:  # NaN
        auroc = accuracy
    if auprc != auprc:  # NaN
        auprc = accuracy

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
    }
