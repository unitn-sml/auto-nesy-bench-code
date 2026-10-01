"""Command-line entry point: ``python -m fonesy <output_dir> <sub-command> ...``."""

import argparse
import logging
import os
import pathlib
import sys
import traceback

import fonesy.general_utils as utils
from fonesy.experiments import compile as compile_exp
from fonesy.experiments import evaluation
from fonesy.experiments import metrics as metrics_exp
from fonesy.experiments import optuna as optuna_exp
from fonesy.experiments import sl as sl_exp
from fonesy.experiments import spl as spl_exp

# Arguments that only exist on `compile`, `spl`, `sl`, `optuna` and `metrics`.
# Removing them from the argument list yields the experiment name of the
# `evaluate` run they operate on (see `args.compile_experiment_name`).
_NON_EVALUATE_KEYS = frozenset((
    # spl / sl
    "spl_task", "backbone", "mlp_hidden", "mlp_layers",
    "spl_hidden", "spl_epochs", "spl_batch_size", "spl_lr", "from_dimacs",
    "auxiliary_num_vars", "sl_weight", "sl_hidden_layers",
    # optuna
    "n_trials", "study_name", "storage",
    "lr_min", "lr_max", "hidden_min", "hidden_max", "batch_sizes", "pruner",
    # metrics
    "epsilon", "delta", "model_count_timeout", "max_exact_vars",
    # compile
    "dimacs", "sdd_out_dir", "aux_vars",
))


def open_samples_sink(args: argparse.Namespace, experiment_name: str):
    """Open ``<experiment>.samples.jsonl`` for ``evaluate`` (``None`` otherwise).

    The file is written after every sample so it survives crashes.
    """
    if args.dry_run or args.func is not evaluation.main:
        return None
    # A new run starts: drop the accepted CNFs of any previous run.
    evaluation._clear_experiment_cnf_artifacts(args.output_dir_path, experiment_name)
    path = args.output_dir_path / (experiment_name + ".samples.jsonl")
    return open(str(path), "w", encoding="utf-8")


def cleanup_incomplete_outputs(args: argparse.Namespace, experiment_name: str) -> None:
    """Remove partial result files after a failure (``.samples.jsonl`` is kept)."""
    for suffix in (".results.json", ".stats.json", ".metrics.csv"):
        p = args.output_dir_path / (experiment_name + suffix)
        if p.exists():
            p.unlink()
            utils.log(f"Removed incomplete output: {p.name}", "WARNING")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="fonesy",
        description="Auto-formalization of neuro-symbolic predictors.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices={"cuda", "cpu"},
        default="cuda",
        help="Which device to run",
    )
    parser.add_argument(
        "output_dir_path",
        metavar="OUTPUT_DIR",
        type=pathlib.Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        default=False,
        help="Don't write any file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=32,
        help="Set the seed for the experiment",
    )

    subparsers = parser.add_subparsers(help="sub-commands help")
    evaluation.configure_subparsers(subparsers)
    metrics_exp.configure_subparsers(subparsers)
    compile_exp.configure_subparsers(subparsers)
    spl_exp.configure_subparsers(subparsers)
    sl_exp.configure_subparsers(subparsers)
    optuna_exp.configure_subparsers(subparsers)

    parsed_args = parser.parse_args()
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)
    return parsed_args


def _add_file_logging(path: pathlib.Path, logger: logging.Logger) -> logging.FileHandler:
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return handler


def main():
    """Main function."""
    args = get_args()
    args.output_dir_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    for noisy in ("matplotlib", "matplotlib.font_manager", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    utils.set_log_level(args.log_level)

    experiment_name = utils.get_basename(args)
    if any(k in vars(args) for k in _NON_EVALUATE_KEYS):
        base_args = argparse.Namespace(
            **{k: v for k, v in vars(args).items() if k not in _NON_EVALUATE_KEYS}
        )
        args.compile_experiment_name = utils.get_basename(base_args)
    else:
        args.compile_experiment_name = None

    if not args.dry_run:
        log_path = args.output_dir_path / (experiment_name + ".log")
        _add_file_logging(log_path, logging.getLogger())
        # <experiment>_think.log: every log line plus the reasoning traces
        # (<think>...</think>) that are stripped from the model outputs.
        think_handler = _add_file_logging(
            args.output_dir_path / (experiment_name + "_think.log"), logging.getLogger()
        )
        think_logger = logging.getLogger("fonesy.think")
        think_logger.propagate = False
        think_logger.setLevel(logging.DEBUG)
        think_logger.addHandler(think_handler)

    utils.log("## START ##", "INFO")
    utils.log("Parsed arguments:", "INFO")
    for arg_name, arg_value in vars(args).items():
        utils.log(f"  {arg_name}: {arg_value}", "INFO")
    utils.log(f"Experiment name: {experiment_name}", "INFO")

    if args.dry_run:
        results_output = open(os.devnull, "wt")
        stats_output = open(os.devnull, "wt")
    else:
        results_output = open(args.output_dir_path / (experiment_name + ".results.json"), "wt", encoding="utf-8")
        stats_output = open(args.output_dir_path / (experiment_name + ".stats.json"), "wt", encoding="utf-8")

    samples_output = open_samples_sink(args, experiment_name)
    args.samples_output_h = samples_output

    utils.set_seed(args.seed)
    device = utils.set_device(args.device)
    utils.log(f"> Seed: {args.seed}, device: {device}", "INFO")

    try:
        args.func(experiment_name, results_output, stats_output, args, device)
    except Exception as exc:
        utils.log(f"## RUN FAILED: {exc} ##", "ERROR")
        utils.log(traceback.format_exc(), "ERROR")
        results_output.close()
        stats_output.close()
        if samples_output is not None:
            samples_output.close()
        if not args.dry_run:
            cleanup_incomplete_outputs(args, experiment_name)
        sys.exit(1)

    results_output.close()
    stats_output.close()
    if samples_output is not None:
        samples_output.close()
    utils.log("## END ##", "INFO")


if __name__ == "__main__":
    main()
