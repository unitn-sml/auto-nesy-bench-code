"""``compile`` sub-command: compile CNFs to SDDs with PySDD.

Batch mode compiles every ``<output_dir>/cnf/<experiment>/*.cnf`` saved by the
matching ``evaluate`` run (same arguments) into ``<output_dir>/sdd/<experiment>/``.
``--dimacs <file>`` compiles a single DIMACS file instead::

    python -m fonesy <output_dir> --seed 32 compile --cnf-format cpmpy \\
        auto-nesy-bench --mode zero-shot-cot vllm --model-path gpt-oss-120b
    python -m fonesy <output_dir> compile \\
        --dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs auto-nesy-bench dummy
"""

from __future__ import annotations

import array
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fonesy.general_utils import log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_subparsers(subparsers):
    """Register the ``compile`` CLI sub-command with all dataset/LLM backends."""
    from fonesy.experiments.evaluation import add_dataset_arguments

    parser = subparsers.add_parser(
        "compile",
        help="Compile saved CNF formulas to SDDs using pysdd",
    )
    add_dataset_arguments(parser)
    parser.add_argument(
        "--dimacs",
        type=Path,
        default=None,
        dest="dimacs",
        help=(
            "Path to a standalone DIMACS file to compile directly, bypassing "
            "the evaluate→compile pipeline (e.g. a ground-truth target)."
        ),
    )
    parser.add_argument(
        "--sdd-out-dir",
        type=Path,
        default=None,
        dest="sdd_out_dir",
        help=(
            "Directory to write the SDD/vtree to when using --dimacs "
            "(default: <output_dir>/sdd/from-dimacs/<stem>/)."
        ),
    )
    parser.add_argument(
        "--aux-vars",
        type=int,
        default=0,
        dest="aux_vars",
        help=(
            "Number of auxiliary (Tseitin) variables at the end of the "
            "--dimacs file's declared variable count (default: 0)."
        ),
    )
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------------
# Auxiliary-variable elimination
# ---------------------------------------------------------------------------
# Tseitin (auxiliary) variables are existentially quantified out of the SDD,
# which is then copied into a manager over the primary variables only, so the
# circuit is defined over the same variables as the ground truth. The copy
# avoids PySDD's vtree surgery, which aborts the process on live variables.


def _copy_sdd_to_manager(root_node: Any, new_manager: Any) -> Any:
    """Copy *root_node* into *new_manager* (iteratively, keeping DAG sharing)."""
    cache: dict[int, Any] = {}
    stack: list[Any] = [root_node]
    while stack:
        node = stack[-1]
        key = node.id
        if key in cache:
            stack.pop()
            continue
        if node.is_false():
            cache[key] = new_manager.false()
            stack.pop()
            continue
        if node.is_true():
            cache[key] = new_manager.true()
            stack.pop()
            continue
        if node.is_literal():
            cache[key] = new_manager.literal(node.literal)
            stack.pop()
            continue
        # Decision node: every (prime, sub) child must be cached first.
        elements = node.elements()
        missing = [
            child for prime, sub in elements for child in (prime, sub)
            if child.id not in cache
        ]
        if missing:
            stack.extend(missing)
            continue
        result = new_manager.false()
        for prime, sub in elements:
            p = cache[prime.id]
            s = cache[sub.id]
            result = new_manager.disjoin(result, new_manager.conjoin(p, s))
        cache[key] = result
        stack.pop()
    return cache[root_node.id]


def _compile_with_auxiliary_elimination(
    cnf_path: Path, primary_num_vars: int, auxiliary_num_vars: int
) -> tuple[Any, Any]:
    """Compile *cnf_path* and eliminate its trailing ``auxiliary_num_vars`` variables.

    Returns ``(manager, root)`` over exactly ``primary_num_vars`` variables.
    """
    from pysdd.sdd import SddManager

    declared_num_vars = primary_num_vars + auxiliary_num_vars
    manager, root = SddManager.from_cnf_file(bytes(str(cnf_path), encoding="utf-8"))
    root.ref()

    exists_map = array.array("i", [0] * (declared_num_vars + 1))
    for v in range(primary_num_vars + 1, declared_num_vars + 1):
        exists_map[v] = 1
    projected = manager.exists_multiple(exists_map, root)
    projected.ref()
    root.deref()
    manager.garbage_collect()

    new_manager = SddManager(var_count=primary_num_vars)
    new_manager.auto_gc_and_minimize_off()
    new_root = _copy_sdd_to_manager(projected, new_manager)
    new_root.ref()
    return new_manager, new_root


# ---------------------------------------------------------------------------
# SDD compilation helpers
# ---------------------------------------------------------------------------


def _compile_one(cnf_path: Path, sdd_dir: Path) -> dict:
    """Compile ``<id>.cnf`` (with its ``.meta.json`` sidecar) to ``<id>.sdd`` / ``<id>.vtree``.

    Returns ``{id, status: "ok" | "error", elapsed_s[, error_message]}``.
    """
    problem_id = cnf_path.stem
    sdd_path = sdd_dir / f"{problem_id}.sdd"
    vtree_path = sdd_dir / f"{problem_id}.vtree"
    sdd_tmp = Path(f"{sdd_path}.tmp.{os.getpid()}")
    vtree_tmp = Path(f"{vtree_path}.tmp.{os.getpid()}")

    # Remove the outputs of a previous compilation first.
    for path in (sdd_path, vtree_path, sdd_tmp, vtree_tmp):
        path.unlink(missing_ok=True)

    t0 = time.monotonic()
    try:
        metadata_path = cnf_path.with_suffix(cnf_path.suffix + ".meta.json")
        if not metadata_path.exists():
            raise ValueError(
                "Missing required CNF metadata sidecar; rerun evaluation so "
                f"{metadata_path.name} is generated before compilation"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("mapping_version") != 1:
            raise ValueError(
                "Invalid CNF metadata sidecar: expected mapping_version=1; "
                "rerun evaluation"
            )
        try:
            primary_num_vars = int(metadata["primary_num_vars"])
            auxiliary_num_vars = int(metadata["auxiliary_num_vars"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid CNF metadata sidecar: primary_num_vars and "
                "auxiliary_num_vars are required integers"
            ) from exc
        if primary_num_vars < 0 or auxiliary_num_vars < 0:
            raise ValueError("Invalid CNF metadata sidecar: variable counts must be nonnegative")

        declared_num_vars = None
        for line in cnf_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[:2] == ["p", "cnf"]:
                declared_num_vars = int(parts[2])
                break
        if declared_num_vars != primary_num_vars + auxiliary_num_vars:
            raise ValueError(
                "CNF metadata/header mismatch: declared variables do not equal "
                "primary_num_vars + auxiliary_num_vars; rerun evaluation"
            )
        if auxiliary_num_vars:
            manager, sdd = _compile_with_auxiliary_elimination(
                cnf_path, primary_num_vars, auxiliary_num_vars
            )
        else:
            from pysdd.sdd import SddManager

            manager, sdd = SddManager.from_cnf_file(bytes(str(cnf_path), encoding='utf-8'))

        # PySDD needs bytes paths; the vtree is published before the SDD.
        sdd.save(bytes(str(sdd_tmp), encoding='utf-8'))
        manager.vtree().save(bytes(str(vtree_tmp), encoding='utf-8'))
        os.replace(vtree_tmp, vtree_path)
        os.replace(sdd_tmp, sdd_path)

        elapsed = time.monotonic() - t0
        log(f"Compiled {problem_id} in {elapsed:.2f}s", "DEBUG")
        return {"id": problem_id, "status": "ok", "elapsed_s": elapsed}

    except Exception as exc:
        for path in (sdd_path, vtree_path, sdd_tmp, vtree_tmp):
            path.unlink(missing_ok=True)
        elapsed = time.monotonic() - t0
        log(f"Compilation failed for {problem_id}: {exc}", "WARNING")
        return {
            "id": problem_id,
            "status": "error",
            "elapsed_s": elapsed,
            "error_message": str(exc),
        }


def compile_dimacs_file(
    cnf_path: Path,
    sdd_dir: Path,
    *,
    primary_num_vars: int | None = None,
    auxiliary_num_vars: int = 0,
) -> dict:
    """Compile a standalone DIMACS file into *sdd_dir*, writing its ``.meta.json``.

    ``primary_num_vars`` defaults to the declared variables minus
    ``auxiliary_num_vars``. Skips compilation when the SDD is newer than the
    source file. Returns the same dict as :func:`_compile_one`.
    """
    sdd_dir.mkdir(parents=True, exist_ok=True)
    problem_id = cnf_path.stem
    sdd_path = sdd_dir / f"{problem_id}.sdd"
    vtree_path = sdd_dir / f"{problem_id}.vtree"
    source_mtime = cnf_path.stat().st_mtime

    if (
        sdd_path.exists()
        and vtree_path.exists()
        and sdd_path.stat().st_mtime >= source_mtime
        and vtree_path.stat().st_mtime >= source_mtime
    ):
        log(f"'{problem_id}' SDD already up to date at {sdd_path} — skipping", "INFO")
        return {"id": problem_id, "status": "ok", "elapsed_s": 0.0, "skipped": True}

    dest_cnf_path = sdd_dir / f"{problem_id}.cnf"
    shutil.copy(cnf_path, dest_cnf_path)

    declared_num_vars = None
    for line in dest_cnf_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[:2] == ["p", "cnf"]:
            declared_num_vars = int(parts[2])
            break
    if declared_num_vars is None:
        raise ValueError(f"Could not find 'p cnf' header in {cnf_path}")

    resolved_primary = (
        primary_num_vars if primary_num_vars is not None
        else declared_num_vars - auxiliary_num_vars
    )
    metadata_path = dest_cnf_path.with_suffix(dest_cnf_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps({
            "mapping_version": 1,
            "primary_num_vars": resolved_primary,
            "auxiliary_num_vars": auxiliary_num_vars,
        }),
        encoding="utf-8",
    )

    return _compile_one(dest_cnf_path, sdd_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Compile all saved CNF files for the given experiment configuration."""
    if getattr(args, "dimacs", None) is not None:
        cnf_source = Path(args.dimacs)
        sdd_dir = args.sdd_out_dir or (
            args.output_dir_path / "sdd" / "from-dimacs" / cnf_source.stem
        )
        log(f"Compiling standalone DIMACS {cnf_source} -> {sdd_dir} ...", "INFO")
        result = compile_dimacs_file(
            cnf_source, sdd_dir, auxiliary_num_vars=args.aux_vars
        )
        n_ok = 1 if result["status"] == "ok" else 0
        n_err = 1 - n_ok
        summary = {
            "experiment": experiment_name,
            "dimacs": str(cnf_source),
            "sdd_dir": str(sdd_dir),
            "n_compiled": n_ok,
            "n_failed": n_err,
            "total_elapsed_s": result["elapsed_s"],
            "results": [result],
        }
        log(
            f"Compilation of {cnf_source.name}: "
            f"{'ok' if n_ok else 'FAILED — ' + result.get('error_message', '')}",
            "INFO",
        )
        json.dump(summary, results_output_h, indent=4)
        json.dump({"n_compiled": n_ok, "n_failed": n_err}, stats_output_h, indent=4)
        return

    # Batch mode reads the CNFs of the evaluate run with the same arguments.
    source_name = getattr(args, "compile_experiment_name", None) or experiment_name
    cnf_dir = args.output_dir_path / "cnf" / source_name
    sdd_dir = args.output_dir_path / "sdd" / source_name

    if not cnf_dir.exists():
        raise FileNotFoundError(
            f"CNF directory not found: {cnf_dir}\n"
            "Run the corresponding 'evaluate' command first."
        )

    if sdd_dir.is_dir():
        stale_paths = set(sdd_dir.glob("*.sdd"))
        stale_paths.update(sdd_dir.glob("*.vtree"))
        stale_paths.update(sdd_dir.glob("*.sdd.tmp.*"))
        stale_paths.update(sdd_dir.glob("*.vtree.tmp.*"))
        for path in stale_paths:
            if path.is_file():
                path.unlink()

    cnf_files = sorted(cnf_dir.glob("*.cnf"))
    log(f"Found {len(cnf_files)} CNF files in {cnf_dir}", "INFO")

    if not cnf_files:
        log("No CNF files to compile — exiting.", "WARNING")
        results = []
    else:
        sdd_dir.mkdir(parents=True, exist_ok=True)

        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_compile_one)(f, sdd_dir) for f in cnf_files
            )
        except ImportError:
            results = [_compile_one(f, sdd_dir) for f in cnf_files]

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    total_time = sum(r["elapsed_s"] for r in results)

    summary = {
        "experiment": experiment_name,
        "n_compiled": n_ok,
        "n_failed": n_err,
        "total_elapsed_s": total_time,
        "sdd_dir": str(sdd_dir),
        "results": results,
    }

    log(
        f"Compilation complete: {n_ok} ok, {n_err} failed, {total_time:.1f}s total",
        "INFO",
    )

    json.dump(summary, results_output_h, indent=4)
    json.dump({"n_compiled": n_ok, "n_failed": n_err}, stats_output_h, indent=4)
