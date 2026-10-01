"""Load the auto-nesy-bench oracle modules (``fonesy/data/auto-nesy-bench/targets/<id>.py``)."""

import importlib.util
import sys
from pathlib import Path

TARGETS_DIR = Path(__file__).resolve().parent.parent / "data" / "auto-nesy-bench" / "targets"


def target_path(problem_id: str, suffix: str = ".dimacs") -> Path:
    """Path of a ground-truth file, e.g. ``target_path("sudoku")`` -> ``.../sudoku.dimacs``."""
    return TARGETS_DIR / f"{problem_id}{suffix}"


def load_oracle(problem_id: str):
    """Import (once) and return the oracle module of *problem_id*."""
    name = f"fonesy.oracles.{problem_id}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, target_path(problem_id, ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]
