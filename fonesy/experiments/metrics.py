"""``metrics`` sub-command: re-score a finished ``evaluate`` run.

``evaluate`` stores every target and final prediction in
``<output_dir>/<experiment>.samples.jsonl``. This command re-verifies those
predictions and recomputes the model-counting metrics without querying the
LLM again, optionally with different ApproxMC parameters. It takes exactly the
same arguments as the ``evaluate`` run it re-scores, e.g.::

    python -m fonesy <output_dir> metrics --cnf-format cpmpy \\
        auto-nesy-bench --mode zero-shot-cot vllm --model-path openai/gpt-oss-120b
"""

import csv
import json
from dataclasses import asdict
from functools import partial

from fonesy.evaluation.dimacs import verify_cnf as verify_dimacs_cnf
from fonesy.evaluation.formula import normalize_variable_names
from fonesy.evaluation.solver import _DEFAULT_DELTA, _DEFAULT_EPSILON, _DEFAULT_TIMEOUT
from fonesy.general_utils import log
from fonesy.statistics.metrics import Metrics
from fonesy.statistics.statistics import CorrectionStatus

_SYMBOLIC_FORMATS = {"nat", "cpmpy", "sympy"}


def load_latest_sample_records(lines):
    """Return one record per sample from the (append-only) samples file.

    A sample whose model count was still running when it was first written
    is appended again once resolved; the last record of each
    ``(batch_idx, sample_idx, id)`` wins and keeps the first position.

    Returns ``(records, source_count, superseded_count)``.
    """
    records, positions, source_count = [], {}, 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        source_count += 1
        batch_idx, sample_idx = record.get("batch_idx"), record.get("sample_idx")
        if batch_idx is None or sample_idx is None:
            records.append(record)
            continue
        identity = (batch_idx, sample_idx, record.get("id"))
        if identity in positions:
            records[positions[identity]] = record
        else:
            positions[identity] = len(records)
            records.append(record)
    return records, source_count, source_count - len(records)


def configure_subparsers(subparsers):
    """Register the ``metrics`` sub-command."""
    from fonesy.experiments.evaluation import add_dataset_arguments

    parser = subparsers.add_parser(
        "metrics", help="Recompute the metrics of a previous evaluate run"
    )
    add_dataset_arguments(parser)
    parser.add_argument("--epsilon", type=float, default=_DEFAULT_EPSILON,
                        help="ApproxMC tolerance.")
    parser.add_argument("--delta", type=float, default=_DEFAULT_DELTA,
                        help="ApproxMC confidence.")
    parser.add_argument("--model-count-timeout", type=int, default=_DEFAULT_TIMEOUT,
                        help="Seconds allowed for each model-count call.")
    parser.add_argument("--max-exact-vars", type=int, default=None,
                        help="Largest variable count counted exactly instead of with ApproxMC.")
    parser.set_defaults(func=main)


def _target_num_vars(record):
    for line in record["target_cnf"].splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[:2] == ["p", "cnf"]:
            return int(parts[2])
    raise ValueError(f"Stored target for sample {record.get('id')!r} has no DIMACS header")


def _reverify(record, cnf_format, primary_num_vars, variable_names):
    """Return ``(pred_cnf, error_message, status)`` for a stored prediction."""
    from fonesy.experiments.evaluation import resolve_format_handlers

    if record.get("predicted_dimacs") is not None:
        # The DIMACS produced by the verifier during evaluation: no need to
        # execute the generated program again.
        return verify_dimacs_cnf(record["predicted_dimacs"], False, primary_num_vars=primary_num_vars)
    if record.get("verification_status") == CorrectionStatus.INVALID.name:
        error = record.get("verification_error") or "[ERROR]: Stored prediction was invalid during evaluation\n"
        return None, error, CorrectionStatus.INVALID
    verify_fn = resolve_format_handlers(cnf_format)["verify_fn"]
    if cnf_format in _SYMBOLIC_FORMATS:
        verify_fn = partial(verify_fn, variable_names=variable_names)
    else:
        verify_fn = partial(verify_fn, primary_num_vars=primary_num_vars)
    return verify_fn(f"<cnf>\n{record['predicted_cnf']}\n</cnf>", True)


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Re-score every sample of the matching ``evaluate`` run."""
    from fonesy.experiments.evaluation import resolve_format_handlers

    source_name = getattr(args, "compile_experiment_name", None) or experiment_name
    samples_path = args.output_dir_path / (source_name + ".samples.jsonl")
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Samples file not found: {samples_path}\nRun the corresponding 'evaluate' command first."
        )

    cnf_format = getattr(args, "cnf_format", "dimacs")
    handlers = resolve_format_handlers(cnf_format)
    metrics = Metrics(
        error_type_codes=handlers["error_codes"],
        classify_errors_fn=handlers["classify_fn"],
        model_count_timeout=args.model_count_timeout,
        epsilon=args.epsilon,
        delta=args.delta,
        max_exact_vars=args.max_exact_vars,
    )

    with open(samples_path, "r", encoding="utf-8") as fh:
        records, n_source, n_superseded = load_latest_sample_records(fh)
    log(f"Loaded {len(records)} samples ({n_superseded} superseded records) from {samples_path}", "INFO")

    per_instance = []
    for record in records:
        record_format = record.get("cnf_format", cnf_format)
        target_num_vars = _target_num_vars(record)
        primary_num_vars = record.get("primary_num_vars") or target_num_vars
        if not (0 <= primary_num_vars <= target_num_vars):
            raise ValueError(f"Invalid primary_num_vars for sample {record.get('id')!r}")

        variable_names = record.get("variable_names")
        if variable_names is not None:
            variable_names = normalize_variable_names(variable_names)[:primary_num_vars]
        elif record_format in _SYMBOLIC_FORMATS:
            raise ValueError(f"Sample {record.get('id')!r} has no stored variable_names")

        pred_cnf, error_message, status = _reverify(
            record, record_format, primary_num_vars, variable_names
        )
        outcome = metrics.update_verified(
            pred_cnf if status != CorrectionStatus.INVALID else None,
            record["target_cnf"],
            error_message=error_message,
            problem_id=record.get("id", ""),
            primary_num_vars=primary_num_vars,
        )
        per_instance.append({
            "id": record.get("id"),
            "cnf_format": record_format,
            "correction_status_after": record.get("correction_status_after"),
            **asdict(outcome),
        })

    aggregate = metrics.compute()
    summary = {
        "experiment": experiment_name,
        "source_experiment": source_name,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "model_count_timeout": args.model_count_timeout,
        "max_exact_vars": args.max_exact_vars,
        "n_instances": len(per_instance),
        "aggregate": aggregate,
        "per_instance": per_instance,
    }
    # default=str: averages of huge model counts may be decimal.Decimal.
    json.dump(summary, results_output_h, indent=4, default=str)
    json.dump(aggregate, stats_output_h, indent=4, default=str)

    if not getattr(args, "dry_run", False):
        csv_path = args.output_dir_path / (experiment_name + ".metrics.csv")
        row = {
            "experiment": experiment_name,
            "source_experiment": source_name,
            "dataset": getattr(args, "dataset", "N/A"),
            "cnf_format": cnf_format,
            "n_instances": len(per_instance),
            **{k: aggregate[k] for k in (
                "syntax_error_rate", "equal_rate", "semantic_error_rate",
                "avg_precision", "avg_recall", "avg_f1", "avg_accuracy",
            )},
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    log("Results written successfully.", "INFO")
