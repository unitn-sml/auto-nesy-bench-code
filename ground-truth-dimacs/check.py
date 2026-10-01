"""Cross-check each auto-nesy-bench DIMACS target against its Python oracle.

An assignment must satisfy ``targets/<id>.dimacs`` iff the oracle
``targets/<id>.py`` accepts it. Tasks with at most 20 variables are checked
exhaustively; larger ones on sampled valid assignments, structured and uniform
random assignments, and single-bit flips of valid assignments.

Usage::

    python ground-truth-dimacs/check.py [task ...] [--samples N]
"""

import argparse
import itertools
import random
import sys
from pathlib import Path

from pysat.formula import CNF
from pysat.solvers import Solver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fonesy.utils.oracles import TARGETS_DIR, load_oracle, target_path  # noqa: E402

EXHAUSTIVE_MAX_VARS = 20


def task_ids():
    return sorted(p.stem for p in TARGETS_DIR.glob("*.dimacs") if not p.stem.endswith("_negated"))


def oracle_accepts(oracle, assignment):
    """Whether the oracle considers the full 0/1 *assignment* valid.

    Concepts outside the task domain (a one-hot concept block that is not
    one-hot, or a binary-encoded digit above 9, on which oracles raise
    ``ValueError``) make the assignment invalid.
    """
    n_concepts = getattr(oracle, "N_CONCEPT_VARS", None)
    layout = (getattr(oracle, "VALIDITY_SPEC", {}) or {}).get("block_layout")
    if n_concepts is not None and layout and sum(layout) == n_concepts:
        start = 0
        for size in layout:
            if sum(assignment[start:start + size]) != 1:
                return False
            start += size
    try:
        if n_concepts is None:
            return bool(oracle.check(list(assignment)))
        expected = [int(bool(v)) for v in oracle.check(list(assignment[:n_concepts]))]
    except ValueError:
        return False
    return list(assignment[n_concepts:]) == expected


def _one_hot_sample(rng, layout, n_vars):
    bits = []
    for size in layout:
        block = [0] * size
        block[rng.randrange(size)] = 1
        bits += block
    return bits + [rng.randint(0, 1) for _ in range(n_vars - len(bits))]


def _positive_sample(rng, oracle, n_vars):
    if hasattr(oracle, "POSITIVE_SAMPLER"):
        return list(oracle.POSITIVE_SAMPLER(rng))
    n_concepts = getattr(oracle, "N_CONCEPT_VARS", None)
    if n_concepts is not None:
        spec = getattr(oracle, "VALIDITY_SPEC", {}) or {}
        layout = spec.get("block_layout")
        concepts = (
            _one_hot_sample(rng, layout, n_concepts) if layout
            else [rng.randint(0, 1) for _ in range(n_concepts)]
        )
        return concepts + [int(bool(v)) for v in oracle.check(concepts)]
    return None


def candidate_assignments(oracle, n_vars, n_samples, seed=0):
    """Assignments to test (all of them when ``n_vars <= EXHAUSTIVE_MAX_VARS``)."""
    if n_vars <= EXHAUSTIVE_MAX_VARS:
        yield from itertools.product((0, 1), repeat=n_vars)
        return
    rng = random.Random(seed)
    layout = (getattr(oracle, "VALIDITY_SPEC", {}) or {}).get("block_layout")
    for _ in range(n_samples):
        positive = _positive_sample(rng, oracle, n_vars)
        if positive is not None:
            yield positive
            flipped = list(positive)
            flipped[rng.randrange(n_vars)] ^= 1
            yield flipped
        if layout and getattr(oracle, "N_CONCEPT_VARS", None) is None:
            yield _one_hot_sample(rng, layout, n_vars)
        yield [rng.randint(0, 1) for _ in range(n_vars)]


def check_task(problem_id, n_samples=500, seed=0):
    """Return ``(n_checked, n_valid, mismatches)`` for one task."""
    oracle = load_oracle(problem_id)
    cnf = CNF(from_file=str(target_path(problem_id)))
    n_vars = oracle.N_VARS
    n_checked = n_valid = 0
    mismatches = []
    with Solver(name="g3", bootstrap_with=cnf.clauses) as solver:
        for assignment in candidate_assignments(oracle, n_vars, n_samples, seed):
            lits = [v if bit else -v for v, bit in enumerate(assignment, start=1)]
            satisfied = solver.solve(assumptions=lits)
            accepted = oracle_accepts(oracle, assignment)
            n_checked += 1
            n_valid += accepted
            if satisfied != accepted:
                mismatches.append(list(assignment))
    return n_checked, n_valid, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    failed = False
    for problem_id in args.tasks or task_ids():
        n_checked, n_valid, mismatches = check_task(problem_id, args.samples)
        status = "ok" if not mismatches else f"{len(mismatches)} MISMATCHES"
        print(f"{problem_id:<14} {n_checked:>9} assignments ({n_valid} valid): {status}")
        failed |= bool(mismatches)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
