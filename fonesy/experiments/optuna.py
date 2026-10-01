"""``optuna`` sub-command: tune the SPL learning rate, head width and batch size.

Uses Optuna's TPE sampler with a median pruner on the per-epoch validation
F1. A trial's objective is the best validation macro-F1 over its epochs. The
circuit and data are built once per study; only the best trial's model is
saved (``<output_dir>/splmodels/<spl-task>/model_optuna_best.pt``). Studies
are stored in ``<output_dir>/optuna_runs/<study-name>.db`` and resumed when
run again::

    python -m fonesy <out> optuna --spl-task sudoku \\
        --from-dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs \\
        --n-trials 30 --spl-epochs 50 auto-nesy-bench dummy
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from fonesy.general_utils import log
from fonesy.experiments.spl import _prepare_spl_run, _train_and_evaluate
from fonesy.spl_utilities.spl_problems import PROBLEM_SPECS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated list of ints, e.g. ``"16,32,64"`` -> ``[16, 32, 64]``."""
    return [int(x) for x in value.split(",")]


def configure_subparsers(subparsers) -> None:
    """Register the ``optuna`` CLI sub-command."""
    from fonesy.experiments.evaluation import add_dataset_arguments

    parser = subparsers.add_parser(
        "optuna",
        help="Hyperparameter-tune an SPL model via Optuna, tracking the best validation-F1 trial",
    )
    # Same arguments as `evaluate`, to locate the SDDs of that run.
    add_dataset_arguments(parser)

    # Same task arguments as `spl`; lr, head width and batch size are searched.
    parser.add_argument(
        "--spl-task",
        choices=sorted(PROBLEM_SPECS.keys()),
        required=True,
        dest="spl_task",
        help="Visual task and logical formula (see fonesy/spl_utilities/spl_problems.py)",
    )
    parser.add_argument(
        "--backbone",
        choices={"conv4", "resnet18", "mlp", "fasterrcnn", "slowfast", "bert"},
        default=None,
        dest="backbone",
        help="Backbone to train from scratch jointly with the head (default: the problem's default_backbone)",
    )
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=512,
        dest="mlp_hidden",
        help="Hidden width of the 'mlp' backbone, fixed across trials (default: 512)",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        choices={2, 3},
        default=3,
        dest="mlp_layers",
        help="Number of layers in the 'mlp' backbone, fixed across trials (default: 3)",
    )
    parser.add_argument(
        "--spl-epochs",
        type=int,
        default=10,
        dest="spl_epochs",
        help="Number of training epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--from-dimacs",
        type=Path,
        default=None,
        dest="from_dimacs",
        help="Path to a DIMACS file to compile on-the-fly and use as the circuit (see spl.py)",
    )
    parser.add_argument(
        "--auxiliary-num-vars",
        type=int,
        default=0,
        dest="auxiliary_num_vars",
        help=(
            "Number of trailing Tseitin/auxiliary variables in the --from-dimacs file "
            "(default: 0). Passed through to compile_dimacs_file so the compiled "
            "circuit's primary scope matches the SPL problem's expected variable count."
        ),
    )

    # ---- Optuna-specific
    parser.add_argument("--n-trials", type=int, default=30, dest="n_trials", help="Number of trials (default: 30)")
    parser.add_argument(
        "--study-name", type=str, default=None, dest="study_name",
        help="Optuna study name (default: optuna-<spl-task>)",
    )
    parser.add_argument(
        "--storage", type=str, default=None, dest="storage",
        help="Optuna storage URL (default: sqlite:///<output_dir>/optuna_runs/<study-name>.db)",
    )
    parser.add_argument("--lr-min", type=float, default=1e-5, dest="lr_min", help="spl_lr search lower bound (default: 1e-5)")
    parser.add_argument("--lr-max", type=float, default=1e-1, dest="lr_max", help="spl_lr search upper bound (default: 1e-1)")
    parser.add_argument("--hidden-min", type=int, default=32, dest="hidden_min", help="spl_hidden search lower bound (default: 32)")
    parser.add_argument("--hidden-max", type=int, default=512, dest="hidden_max", help="spl_hidden search upper bound (default: 512)")
    parser.add_argument(
        "--batch-sizes", type=_parse_int_csv, default=[16, 32, 64, 128, 256], dest="batch_sizes",
        help=(
            "spl_batch_size categorical search values, comma-separated "
            "(default: 16,32,64,128,256). Not nargs='+' since the dataset/LLM "
            "positional args always follow all flags on this CLI."
        ),
    )
    parser.add_argument(
        "--pruner", choices={"median", "none"}, default="median", dest="pruner",
        help="Optuna pruner: 'median' stops clearly-bad trials early using intermediate val_f1 (default: median)",
    )
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------


def _suggest_trial_args(base_args: Any, trial: Any) -> Any:
    """Copy of *base_args* with the hyperparameters sampled by *trial*."""
    trial_args = copy.copy(base_args)
    trial_args.spl_lr = trial.suggest_float("spl_lr", base_args.lr_min, base_args.lr_max, log=True)
    trial_args.spl_hidden = trial.suggest_int("spl_hidden", base_args.hidden_min, base_args.hidden_max)
    trial_args.spl_batch_size = trial.suggest_categorical("spl_batch_size", base_args.batch_sizes)
    return trial_args


def _make_objective(spec, prepared: dict, base_args: Any, device: torch.device, best_state: dict):
    import optuna

    def objective(trial: "optuna.trial.Trial") -> float:
        trial_args = _suggest_trial_args(base_args, trial)

        def epoch_callback(epoch: int, entry: dict) -> None:
            trial.report(entry["val_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            run = _train_and_evaluate(
                prepared, spec, trial_args, device, epoch_callback=epoch_callback,
            )
        except optuna.TrialPruned:
            raise
        except (ValueError, RuntimeError) as exc:
            # Diverging configurations (NaN weights) are pruned.
            log(
                f"Trial {trial.number} diverged/errored with params="
                f"{ {'spl_lr': trial_args.spl_lr, 'spl_hidden': trial_args.spl_hidden, 'spl_batch_size': trial_args.spl_batch_size} }: "
                f"{exc!r} — pruning trial.",
                "WARNING",
            )
            raise optuna.TrialPruned() from exc
        best_trial_val_f1 = max((r["val_f1"] for r in run["history"]), default=0.0)

        if best_trial_val_f1 > best_state["value"]:
            best_state["value"] = best_trial_val_f1
            best_state["trial_number"] = trial.number
            best_state["params"] = {
                "spl_lr": trial_args.spl_lr,
                "spl_hidden": trial_args.spl_hidden,
                "spl_batch_size": trial_args.spl_batch_size,
            }
            best_state["run"] = run
            log(
                f"Trial {trial.number}: new best val_f1={best_trial_val_f1:.4f} "
                f"(params={best_state['params']})",
                "INFO",
            )

        return best_trial_val_f1

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    experiment_name: str,
    results_output: Any,
    stats_output: Any,
    args: Any,
    device: torch.device,
) -> None:
    """Run the Optuna study for ``--spl-task`` and save the best model."""
    import optuna

    spec = PROBLEM_SPECS[args.spl_task]

    prepared = _prepare_spl_run(spec, args, device, experiment_name)

    study_name = args.study_name or f"optuna-{args.spl_task}"
    storage = args.storage
    if storage is None:
        db_dir = args.output_dir_path / "optuna_runs"
        db_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_dir / study_name}.db"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner() if args.pruner == "median" else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    best_state: dict = {"value": -1.0, "trial_number": None, "params": None, "run": None}
    objective = _make_objective(spec, prepared, args, device, best_state)

    log(
        f"Starting Optuna study '{study_name}' ({args.n_trials} trials, "
        f"{args.spl_epochs} epochs/trial) for spl_task={args.spl_task} -> {storage}",
        "INFO",
    )
    study.optimize(objective, n_trials=args.n_trials)

    log(
        f"Best trial: #{study.best_trial.number} val_f1={study.best_value:.4f} "
        f"params={study.best_params}",
        "INFO",
    )

    # ---- Save the best trial's already-trained model
    model_path = None
    if not args.dry_run and best_state["run"] is not None:
        best_run = best_state["run"]
        save_dir = args.output_dir_path / "splmodels" / args.spl_task
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model_optuna_best.pt"
        torch.save(
            {
                "head":              best_run["head"].state_dict(),
                "backbone_features": (
                    best_run["backbone"].features.state_dict()
                    if best_run["backbone"] is not None else None
                ),
                "backbone_name":     best_run["backbone_name"],
                "spl_task":          args.spl_task,
                "spl_hidden":        best_state["params"]["spl_hidden"],
                "feature_dim":       best_run["feature_dim"],
                "gf_size":           best_run["gf_size"],
                "gf_specs":          {k: list(v) for k, v in prepared["gf_specs"].items()},
                "best_trial_number": best_state["trial_number"],
                "best_params":       best_state["params"],
            },
            model_path,
        )
        log(f"Best SPL model (trial #{best_state['trial_number']}) saved to {model_path}", "INFO")
    elif args.dry_run:
        log("Dry-run: skipping best-model save", "INFO")

    # ---- Write results / stats JSON
    trials_summary = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]
    results = {
        "experiment":  experiment_name,
        "spl_task":    args.spl_task,
        "study_name":  study_name,
        "storage":     storage,
        "n_trials":    args.n_trials,
        "epochs_per_trial": args.spl_epochs,
        "search_space": {
            "spl_lr":         [args.lr_min, args.lr_max],
            "spl_hidden":     [args.hidden_min, args.hidden_max],
            "spl_batch_size": args.batch_sizes,
        },
        "trials":              trials_summary,
        "best_trial_number":   study.best_trial.number,
        "best_val_f1":         study.best_value,
        "best_params":         study.best_params,
        "model_path":          str(model_path) if model_path is not None else None,
    }
    json.dump(results, results_output, indent=4)

    _formula_rel = prepared.get("formula_rel") or {"formula_rel": "n/a", "mc_ratio": None}
    stats = {
        "best_val_f1":       study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_spl_lr":         study.best_params.get("spl_lr"),
        "best_spl_hidden":     study.best_params.get("spl_hidden"),
        "best_spl_batch_size": study.best_params.get("spl_batch_size"),
        "formula_rel":       _formula_rel["formula_rel"],
        "formula_mc_ratio":  _formula_rel["mc_ratio"],
    }
    if best_state["run"] is not None:
        stats.update({f"best_trial_test_{k}": v for k, v in best_state["run"]["test_metrics"].items()})
    json.dump(stats, stats_output, indent=4)
