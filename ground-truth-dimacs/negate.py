"""Precompute the Tseitin negation of every auto-nesy-bench target.

Model counting FP = MC(~target & pred) needs the negated target. For large
targets (``kand-logic`` has 129,648 clauses) the negation is written once to
``targets/<id>_negated.dimacs`` and reused (see
``fonesy.evaluation.solver.load_precomputed_negation``) instead of being
recomputed for every sample. The files are optional: without them the target
is negated on the fly, with identical results.

Usage::

    python ground-truth-dimacs/negate.py [--targets-dir DIR]
"""

import argparse
import sys
import time
from pathlib import Path

from pysat.formula import CNF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fonesy.evaluation.solver import _declared_num_vars  # noqa: E402
from fonesy.utils.oracles import TARGETS_DIR  # noqa: E402

PROGRAM_NAMES = sorted(
    p.stem for p in TARGETS_DIR.glob("*.dimacs") if not p.stem.endswith("_negated")
)


def negate_target_dimacs(target_str: str) -> str:
    """Negate a DIMACS formula exactly as ``solver._build_confusion_dimacs`` does.

    The ``c source_nv`` / ``c source_clauses`` comments let the solver detect a
    negation computed from an outdated target.
    """
    target = CNF(from_string=target_str)
    nv = _declared_num_vars(target_str)
    target.nv = nv
    not_target = target.negate(topv=nv)
    return f"c source_nv: {nv}\nc source_clauses: {len(target.clauses)}\n{not_target.to_dimacs()}"


def negate_all_program_targets(targets_dir: Path = TARGETS_DIR) -> dict:
    """Return ``{id: negated DIMACS}`` for every target in *targets_dir*."""
    return {
        name: negate_target_dimacs((Path(targets_dir) / f"{name}.dimacs").read_text())
        for name in PROGRAM_NAMES
    }


def write_negated_targets(targets_dir: Path = TARGETS_DIR) -> dict:
    """Write ``<id>_negated.dimacs`` next to each target; return ``{id: path}``."""
    targets_dir = Path(targets_dir)
    outputs = {}
    for name in PROGRAM_NAMES:
        t0 = time.monotonic()
        out_path = targets_dir / f"{name}_negated.dimacs"
        out_path.write_text(negate_target_dimacs((targets_dir / f"{name}.dimacs").read_text()))
        outputs[name] = out_path
        print(f"{name:<14} {time.monotonic() - t0:6.2f}s -> {out_path}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets-dir", default=str(TARGETS_DIR))
    write_negated_targets(Path(parser.parse_args().targets_dir))
