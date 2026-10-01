"""Semantic loss baseline: backbone + MLP classifier trained with
BCE + w * SL, where SL = -log P(constraint) is computed on the compiled SDD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fonesy.experiments.spl import (
    _DATA_DIR,
    _fmt_metrics,
    _get_feature_dim,
    _save_confusion_matrix_pdf,
)
from fonesy.general_utils import log
from fonesy.spl_utilities.spl_metrics import (
    build_ground_truth_masks,
    compute_classification_metrics,
    is_consistent,
)
from fonesy.spl_utilities.spl_problems import PROBLEM_SPECS, ProblemSpec
from fonesy.utils.factories import NetworkFactory, TaskFactory


def configure_subparsers(subparsers) -> None:
    from fonesy.experiments.evaluation import add_dataset_arguments

    parser = subparsers.add_parser("sl", help="Train a classifier with the semantic loss")
    add_dataset_arguments(parser)
    parser.add_argument("--spl-task", choices=sorted(PROBLEM_SPECS.keys()), required=True, dest="spl_task")
    parser.add_argument("--backbone", default=None, dest="backbone",
                        choices={"conv4", "resnet18", "mlp", "fasterrcnn", "slowfast", "bert"})
    parser.add_argument("--mlp-hidden", type=int, default=512, dest="mlp_hidden")
    parser.add_argument("--mlp-layers", type=int, choices={2, 3}, default=3, dest="mlp_layers")
    parser.add_argument("--spl-hidden", type=int, default=256, dest="spl_hidden")
    parser.add_argument("--spl-epochs", type=int, default=10, dest="spl_epochs")
    parser.add_argument("--spl-batch-size", type=int, default=256, dest="spl_batch_size")
    parser.add_argument("--spl-lr", type=float, default=1e-3, dest="spl_lr")
    parser.add_argument("--sl-hidden-layers", type=int, default=3, dest="sl_hidden_layers")
    parser.add_argument("--sl-weight", type=float, default=1.0, dest="sl_weight")
    parser.add_argument("--from-dimacs", type=Path, default=None, dest="from_dimacs")
    parser.add_argument("--auxiliary-num-vars", type=int, default=0, dest="auxiliary_num_vars")
    parser.set_defaults(func=main)


def _locate_sdd(spec: ProblemSpec, args: Any, experiment_name: str) -> Path:
    if args.from_dimacs is not None:
        from fonesy.experiments.compile import compile_dimacs_file

        sdd_dir = args.output_dir_path / "sdd" / "from-dimacs" / spec.problem_id
        result = compile_dimacs_file(Path(args.from_dimacs), sdd_dir,
                                     auxiliary_num_vars=args.auxiliary_num_vars or 0)
        if result["status"] != "ok":
            raise RuntimeError(f"DIMACS compilation failed: {result.get('error_message')}")
        return sdd_dir / f"{spec.problem_id}.sdd"
    name = getattr(args, "compile_experiment_name", None) or experiment_name
    sdd_path = args.output_dir_path / "sdd" / name / f"{spec.problem_id}.sdd"
    if not sdd_path.exists():
        raise FileNotFoundError(f"SDD not found: {sdd_path}")
    return sdd_path


def _build_constraint(sdd_path: Path, device: torch.device) -> nn.Module:
    import klay

    circuit = klay.Circuit()
    circuit.add_sdd_from_file(str(sdd_path))
    # compile=False: the last batch has a different shape.
    return circuit.to_torch_module(semiring="log", compile=False).to(device)


def semantic_loss(logits: torch.Tensor, constraint: nn.Module) -> torch.Tensor:
    """Per-sample -log P(constraint), variables independent Bernoulli(sigmoid(logit))."""
    return -constraint(F.logsigmoid(logits), F.logsigmoid(-logits))[:, -1]


def _build_backbone(name, task, args, device):
    if name is None:
        return None, task.feature_dim
    kwargs = dict(n_classes=task.n_classes, in_channels=task.in_channels)
    if name == "mlp":
        kwargs.update(in_dim=task.feature_dim, hidden_dim=args.mlp_hidden, n_layers=args.mlp_layers)
    else:
        kwargs.update(image_size=task.image_size)
    backbone = NetworkFactory.get_network(name, **kwargs).to(device)
    return backbone, _get_feature_dim(backbone)


def _build_classifier(in_dim: int, hidden: int, n_layers: int, n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden if n_layers else in_dim, n_out))
    return nn.Sequential(*layers)


def _logits(backbone, classifier, batch, device):
    x, raw = batch[0].to(device), batch[1].to(device)
    feats = backbone.features(x).flatten(1) if backbone is not None else x
    return classifier(feats), raw


def _class_probs(logits: torch.Tensor, spec: ProblemSpec) -> torch.Tensor:
    """P(output bits = pattern of class k), normalised over the classes."""
    out = logits[:, spec.n_concept_vars : spec.n_vars]
    patterns = torch.stack([spec.class_to_output_bits(k) for k in range(spec.n_classes)]).to(logits.device)
    log_p = F.logsigmoid(out) @ patterns.T + F.logsigmoid(-out) @ (1 - patterns).T
    return log_p.softmax(dim=-1)


def _train_epoch(backbone, classifier, constraint, spec, loader, optimizer, device, sl_weight):
    classifier.train()
    if backbone is not None:
        backbone.train()
    total = total_loss = total_sl = 0.0
    for batch in loader:
        logits, raw = _logits(backbone, classifier, batch, device)
        target = spec.encode_x_truth(raw)
        # BCE summed over variables, to match the per-sample scale of SL.
        sup = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1).mean()
        sl = semantic_loss(logits, constraint).mean()
        loss = sup + sl_weight * sl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = logits.size(0)
        total, total_loss, total_sl = total + n, total_loss + loss.item() * n, total_sl + sl.item() * n
    return (total_loss / total, total_sl / total) if total else (0.0, 0.0)


@torch.no_grad()
def _eval_epoch(backbone, classifier, constraint, spec, loader, device, gt_masks=None):
    classifier.eval()
    if backbone is not None:
        backbone.eval()
    y_true, y_pred, y_score, consistent, sl = [], [], [], [], []
    for batch in loader:
        logits, raw = _logits(backbone, classifier, batch, device)
        target = spec.encode_x_truth(raw)
        states = (logits > 0).float()
        if spec.output_kind == "multi_label":
            pred = spec.decode_downstream(states[:, spec.eval_start_var : spec.n_vars])
            scores = torch.sigmoid(logits[:, spec.eval_start_var : spec.n_vars])
        else:
            scores = _class_probs(logits, spec)
            pred = scores.argmax(dim=-1)
        y_true.append(spec.decode_downstream(target[:, spec.eval_start_var : spec.n_vars]).cpu())
        y_pred.append(pred.cpu())
        y_score.append(scores.cpu())
        sl.append(semantic_loss(logits, constraint).cpu())
        if gt_masks is not None:
            consistent.append(is_consistent(states, *gt_masks).cpu())

    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    metrics = compute_classification_metrics(y_true, y_pred, torch.cat(y_score).numpy(), spec)
    if gt_masks is not None:
        flags = torch.cat(consistent)
        metrics.update(consistency=flags.float().mean().item(), n_predictions=int(flags.numel()),
                       n_violations=int((~flags).sum().item()))
    metrics["semantic_loss"] = torch.cat(sl).mean().item()
    return metrics, y_true, y_pred


def main(experiment_name: str, results_output: Any, stats_output: Any, args: Any, device: torch.device) -> None:
    spec = PROBLEM_SPECS[args.spl_task]
    sdd_path = _locate_sdd(spec, args, experiment_name)
    task = TaskFactory.get_task(spec.task_name, root=str(_DATA_DIR))
    constraint = _build_constraint(sdd_path, device)

    gt_dimacs = _DATA_DIR / "auto-nesy-bench" / "targets" / f"{spec.problem_id}.dimacs"
    try:
        gt_masks = build_ground_truth_masks(gt_dimacs, spec.n_vars, device)
    except ValueError as exc:
        log(f"Skipping consistency metric: {exc}", "WARNING")
        gt_masks = None

    backbone_name = args.backbone or spec.default_backbone
    backbone, feature_dim = _build_backbone(backbone_name, task, args, device)
    classifier = _build_classifier(feature_dim, args.spl_hidden, args.sl_hidden_layers, spec.n_vars).to(device)
    params = list(classifier.parameters()) + (list(backbone.features.parameters()) if backbone is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.spl_lr)

    train_loader = task.train_loader(args.spl_batch_size)
    val_loader = task.val_loader(args.spl_batch_size)
    test_loader = task.test_loader(args.spl_batch_size)

    history = []
    for epoch in range(1, args.spl_epochs + 1):
        tr_loss, tr_sl = _train_epoch(backbone, classifier, constraint, spec, train_loader,
                                      optimizer, device, args.sl_weight)
        val, _, _ = _eval_epoch(backbone, classifier, constraint, spec, val_loader, device, gt_masks)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_semantic_loss": tr_sl,
                        **{f"val_{k}": v for k, v in val.items()}})
        log(f"Epoch {epoch}/{args.spl_epochs}: train_loss={tr_loss:.4f} train_sl={tr_sl:.4f} "
            f"{_fmt_metrics(val)}", "INFO")

    test, test_true, test_pred = _eval_epoch(backbone, classifier, constraint, spec, test_loader, device, gt_masks)
    log("Test: " + _fmt_metrics(test), "INFO")

    model_path = args.output_dir_path / "slmodels" / args.spl_task / "model.pt"
    if not args.dry_run:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "classifier": classifier.state_dict(),
            "backbone_features": backbone.features.state_dict() if backbone is not None else None,
            "backbone_name": backbone_name, "spl_task": args.spl_task, "spl_hidden": args.spl_hidden,
            "sl_hidden_layers": args.sl_hidden_layers, "sl_weight": args.sl_weight,
            "feature_dim": feature_dim, "n_vars": spec.n_vars,
        }, model_path)

    json.dump({
        "experiment": experiment_name, "spl_task": args.spl_task, "backbone": backbone_name,
        "sdd_path": str(sdd_path), "model_path": str(model_path), "epochs": args.spl_epochs,
        "sl_weight": args.sl_weight, "sl_hidden_layers": args.sl_hidden_layers,
        "training": history, "test": test,
    }, results_output, indent=4)

    best = max(history, key=lambda r: r["val_accuracy"]) if history else {}
    json.dump({
        "best_val_accuracy": best.get("val_accuracy"), "best_val_f1": best.get("val_f1"),
        "best_val_epoch": best.get("epoch"), **{f"final_test_{k}": v for k, v in test.items()},
    }, stats_output, indent=4)

    if not args.dry_run and test_true.size:
        plot_dir = args.output_dir_path / "slplots" / args.spl_task
        plot_dir.mkdir(parents=True, exist_ok=True)
        if spec.output_kind != "multi_label":
            _save_confusion_matrix_pdf(test_true, test_pred, args.spl_task,
                                       list(range(spec.n_classes)), plot_dir / f"{args.spl_task}_cm.pdf")
        elif len(spec.output_names) <= 12:
            for j, name in enumerate(spec.output_names):
                _save_confusion_matrix_pdf(test_true[:, j], test_pred[:, j], f"{name} - {args.spl_task}",
                                           [0, 1], plot_dir / f"{name}_cm.pdf")
