r"""``spl`` sub-command: train a Semantic Probabilistic Layer (SPL) predictor.

A backbone and an MLP head are trained jointly; the head outputs the sum-layer
parameters of a circuit compiled from a CNF, so predictions always satisfy the
CNF. Training minimizes the negative log-likelihood of the full ground-truth
assignment; evaluation uses MAP inference (accuracy, precision, recall, F1),
marginals (AUROC, AUPRC), consistency with the ground-truth formula and, for
``--from-dimacs``, the relation (REL) and model-count ratio (MC-R) between
the formula and the ground truth.

The circuit comes either from ``--from-dimacs <file>`` (e.g. a ground-truth
target or an LLM-generated CNF) or from the SDDs written by ``compile`` for the
``evaluate`` run with the same arguments::

    python -m fonesy <out> spl --spl-task sudoku \
        --from-dimacs fonesy/data/auto-nesy-bench/targets/sudoku.dimacs \
        --spl-epochs 50 --spl-lr 2.9e-4 --spl-batch-size 16 --spl-hidden 455 \
        auto-nesy-bench dummy
"""

from __future__ import annotations

import json
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fonesy.general_utils import log
from fonesy.spl_utilities.spl_metrics import (
    build_ground_truth_masks,
    compute_classification_metrics,
    compute_marginal_probs,
    formula_vs_oracle,
    is_consistent,
    make_integrate_query,
)
from fonesy.spl_utilities.spl_problems import PROBLEM_SPECS, ProblemSpec
from fonesy.utils.factories import NetworkFactory, TaskFactory


_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_subparsers(subparsers) -> None:
    """Register the ``spl`` CLI sub-command."""
    from fonesy.experiments.evaluation import add_dataset_arguments

    parser = subparsers.add_parser(
        "spl",
        help="Train an SPL (backbone + head, from scratch) against a ground-truth or LLM-predicted circuit",
    )
    # Same arguments as `evaluate`, to locate the SDDs of that run.
    add_dataset_arguments(parser)

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
        help=(
            "Backbone to train from scratch jointly with the head "
            "(default: the problem's default_backbone — 'mlp' for bdd-oia[-simple], "
            "whose precomputed features are first passed through a small MLP; "
            "'bert' for cebab, a frozen bert-base-uncased + trained MLP projection)"
        ),
    )
    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=512,
        dest="mlp_hidden",
        help="Hidden width of the 'mlp' backbone (default: 512)",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        choices={2, 3},
        default=3,
        dest="mlp_layers",
        help="Number of layers in the 'mlp' backbone (default: 3)",
    )
    parser.add_argument(
        "--spl-hidden",
        type=int,
        default=256,
        dest="spl_hidden",
        help="Hidden units in the MLP parameter head (default: 256)",
    )
    parser.add_argument(
        "--spl-epochs",
        type=int,
        default=10,
        dest="spl_epochs",
        help="Number of SPL training epochs (default: 10)",
    )
    parser.add_argument(
        "--spl-batch-size",
        type=int,
        default=256,
        dest="spl_batch_size",
        help="Mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--spl-lr",
        type=float,
        default=1e-3,
        dest="spl_lr",
        help="Adam learning rate for backbone+head (default: 0.001)",
    )
    parser.add_argument(
        "--from-dimacs",
        type=Path,
        default=None,
        dest="from_dimacs",
        help=(
            "Path to a DIMACS file to compile on-the-fly and use as the circuit, "
            "bypassing the evaluate→compile pipeline.  Supports both ground-truth "
            "and LLM-generated DIMACS files."
        ),
    )
    parser.add_argument(
        "--auxiliary-num-vars",
        type=int,
        default=0,
        dest="auxiliary_num_vars",
        help=(
            "Number of trailing Tseitin/auxiliary variables in the --from-dimacs file "
            "(default: 0). When non-zero, primary_num_vars is set to "
            "declared_num_vars - auxiliary_num_vars so the compiled circuit's scope "
            "matches the SPL problem's expected variable count."
        ),
    )
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------------
# Circuit helpers (libcirkit 0.2.x compatible)
# ---------------------------------------------------------------------------


def _make_map_query(circuit: Any) -> Any:
    """Instantiate a MAPQuery for *circuit* (local import keeps cirkit optional)."""
    from cirkit.backend.torch.queries import MAPQuery
    return MAPQuery(circuit)


def _canonical_layer_order(symbolic: Any) -> list:
    """Deterministic order of the layers of a symbolic circuit.

    cirkit builds the layer list from a set, so its order (and therefore the
    mapping from head outputs to sum layers) changes between runs and a saved
    model could not be reloaded. Layers are coloured by Weisfeiler-Leman
    refinement (seeded with their variable scope) and emitted by a DFS from the
    output visiting inputs in colour order.
    """
    layers = list(symbolic.layers)
    inputs_of = {l: list(symbolic.layer_inputs(l)) for l in layers}
    parents_of: dict[Any, list] = {l: [] for l in layers}
    for l, ins in inputs_of.items():
        for src in ins:
            parents_of[src].append(l)

    # reachable variable scope, bottom-up over a topological pass
    indeg = {l: len(inputs_of[l]) for l in layers}
    ready = [l for l in layers if indeg[l] == 0]
    topo: list = []
    while ready:
        l = ready.pop()
        topo.append(l)
        for succ in parents_of[l]:
            indeg[succ] -= 1
            if indeg[succ] == 0:
                ready.append(succ)
    scope: dict[Any, frozenset] = {}
    for l in topo:
        s: set = set(getattr(l, "scope", []) or [])
        for c in inputs_of[l]:
            s |= scope[c]
        scope[l] = frozenset(s)

    def base_key(l: Any) -> tuple:
        return (
            type(l).__name__,
            int(getattr(l, "arity", -1)),
            int(getattr(l, "num_input_units", -1)),
            int(getattr(l, "num_output_units", -1)),
            tuple(sorted(scope[l])),
        )

    colour = {l: base_key(l) for l in layers}
    for _ in range(len(layers) + 1):
        new = {
            l: (
                colour[l],
                tuple(sorted(colour[c] for c in inputs_of[l])),
                tuple(sorted(colour[p] for p in parents_of[l])),
            )
            for l in layers
        }
        # compress to ranks so tuples don't grow unboundedly
        ranks = {c: i for i, c in enumerate(sorted(set(new.values())))}
        new = {l: (ranks[v],) for l, v in new.items()}
        if len(set(new.values())) == len(set(colour.values())) and new == colour:
            break
        colour = new

    sort_key = {l: (colour[l], tuple(sorted(scope[l]))) for l in layers}
    children_sorted = {
        l: sorted(inputs_of[l], key=lambda x: sort_key[x]) for l in layers
    }

    order: list = []
    seen: set = set()
    for out in symbolic.outputs:  # single output for an SDD, but be general
        if out in seen:
            continue
        stack = [(out, False)]
        while stack:  # iterative post-order DFS (circuits get deep — no recursion)
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for c in reversed(children_sorted[node]):
                if c not in seen:
                    stack.append((c, False))

    if len(order) != len(layers):  # unreachable layers — append deterministically
        order.extend(sorted((l for l in layers if l not in seen),
                            key=lambda l: (colour[l], tuple(sorted(scope[l])))))
    return order


def _build_circuit(sdd_path: Path, vtree_path: Path) -> tuple[Any, dict]:
    """Load an SDD as a torch circuit whose sum layers are set by gate functions.

    Returns ``(circuit, gf_specs)`` where ``gf_specs`` maps each gate-function
    key to the shape of its parameters (the head output is split accordingly).
    """
    from cirkit.templates.logic import SDD
    from cirkit.pipeline import PipelineContext
    from cirkit.symbolic.circuit import Circuit
    from cirkit.symbolic.functional import condition_circuit

    symbolic = SDD.from_file(str(sdd_path)).build_circuit(sum_weight_activation="softmax")

    ordered = _canonical_layer_order(symbolic)
    symbolic = Circuit(
        ordered,
        {l: list(symbolic.layer_inputs(l)) for l in ordered},
        list(symbolic.outputs),
    )

    parametrization_map = {"sum": list(symbolic.sum_layers)}
    conditional_symbolic, gf_specs = condition_circuit(
        symbolic, gate_functions=parametrization_map
    )

    ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True)
    for gf_key, shape in gf_specs.items():
        ctx.add_gate_function(gf_key, _make_reshape_fn(shape))
    circuit = ctx.compile(conditional_symbolic)
    return circuit, gf_specs


def _make_reshape_fn(shape: tuple[int, ...]):
    """Return a closure reshaping a flat head-output chunk to *shape* (per gate function)."""
    def reshape(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, *shape)
    return reshape


def _total_gate_param_size(gf_specs: dict) -> int:
    """Total flat head-output size needed to cover every gate function in *gf_specs*."""
    return sum(reduce(mul, shape) for shape in gf_specs.values())


def _split_gate_function_kwargs(gate_params: torch.Tensor, gf_specs: dict) -> dict:
    """Split a flat ``(B, total_gf_size)`` head output into per-gate-function chunks.

    ``gf_specs`` order is preserved (Python dicts are insertion-ordered),
    matching the order the head's final Linear layer was sized in.
    """
    kwargs = {}
    offset = 0
    for key, shape in gf_specs.items():
        size = reduce(mul, shape)
        kwargs[key] = {"x": gate_params[:, offset : offset + size]}
        offset += size
    return kwargs


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _get_feature_dim(network: nn.Module) -> int:
    """Size of ``network.features``' output (input size of its classifier)."""
    c = network.classifier
    if isinstance(c, nn.Linear):
        return c.in_features
    for layer in c:
        if isinstance(layer, nn.Linear):
            return layer.in_features
    raise ValueError("Cannot determine feature dimension from network.classifier")


def _forward_gate_params(
    backbone: nn.Module | None, head: nn.Module, batch: tuple, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(head output, raw labels)`` for one batch."""
    model_input = batch[0].to(device)
    raw_concepts = batch[1].to(device)
    if backbone is not None:
        feats = backbone.features(model_input).flatten(1)
    else:
        feats = model_input
    gate_params = head(feats)
    return gate_params, raw_concepts


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _train_epoch(
    backbone: nn.Module | None,
    head: nn.Module,
    circuit: Any,
    gf_specs: dict,
    spec: ProblemSpec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One epoch minimizing the NLL of the full ground-truth assignment."""
    head.train()
    if backbone is not None:
        backbone.train()

    total_loss = 0.0
    total = 0

    for batch in loader:
        gate_params, raw_concepts = _forward_gate_params(backbone, head, batch, device)
        x_truth = spec.encode_x_truth(raw_concepts)
        gate_function_kwargs = _split_gate_function_kwargs(gate_params, gf_specs)

        likelihoods = circuit(
            x_truth,
            gate_function_kwargs=gate_function_kwargs,
        )
        loss = -likelihoods.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * gate_params.size(0)
        total += gate_params.size(0)

    return total_loss / total if total else 0.0


def _eval_epoch(
    backbone: nn.Module | None,
    head: nn.Module,
    map_query: Any,
    integrate_query: Any,
    gf_specs: dict,
    spec: ProblemSpec,
    loader: DataLoader,
    device: torch.device,
    gt_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict, "np.ndarray", "np.ndarray"]:
    """Evaluate with MAP inference and circuit marginals.

    With *gt_masks* (see ``spl_metrics.build_ground_truth_masks``) also reports
    ``consistency``: the fraction of MAP assignments that satisfy the
    ground-truth formula. Returns ``(metrics, y_true, y_pred)``.
    """
    head.eval()
    if backbone is not None:
        backbone.eval()

    all_true, all_pred, all_score, all_consistent = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            gate_params, raw_concepts = _forward_gate_params(backbone, head, batch, device)
            gate_function_kwargs = _split_gate_function_kwargs(gate_params, gf_specs)

            x_truth = spec.encode_x_truth(raw_concepts)
            true_label = spec.decode_downstream(x_truth[:, spec.eval_start_var : spec.n_vars])

            _, pred_states = map_query(gate_function_kwargs=gate_function_kwargs)
            scores = compute_marginal_probs(integrate_query, gate_function_kwargs, spec, device)

            if spec.output_kind == "multi_label":
                pred_output_bits = pred_states[:, spec.eval_start_var : spec.n_vars].float()
                pred_label = spec.decode_downstream(pred_output_bits)
            else:
                # Multiclass predictions are the argmax of the class marginals: joint MAP
                # decoding can collapse onto a single input-independent mode.
                pred_label = scores.argmax(dim=-1)

            all_true.append(true_label.cpu())
            all_pred.append(pred_label.cpu())
            all_score.append(scores.cpu())

            if gt_masks is not None:
                all_consistent.append(is_consistent(pred_states.float(), *gt_masks).cpu())

    if not all_true:
        empty = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.0, "auprc": 0.0}
        if gt_masks is not None:
            empty["consistency"] = 1.0
            empty["n_predictions"] = 0
            empty["n_violations"] = 0
        return empty, torch.empty(0).numpy(), torch.empty(0).numpy()

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_score = torch.cat(all_score).numpy()

    metrics = compute_classification_metrics(y_true, y_pred, y_score, spec)
    if gt_masks is not None:
        flags = torch.cat(all_consistent)  # (N,) bool
        metrics["consistency"] = flags.float().mean().item()
        metrics["n_predictions"] = int(flags.numel())
        metrics["n_violations"] = int((~flags).sum().item())
    return metrics, y_true, y_pred


def _save_confusion_matrix_pdf(
    true_labels,
    pred_labels,
    title: str,
    labels: list,
    save_path: Path,
) -> None:
    """Save a confusion matrix to *save_path* (PDF), annotating non-zero cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n  = len(labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    fig_w = max(6, n * 0.6) + 1.2   # extra for colorbar
    fig_h = max(5, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11)

    thresh    = cm.max() / 2.0
    ann_size  = max(5, 9 - n // 5)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v > 0:
                ax.text(
                    j, i, str(v),
                    ha="center", va="center",
                    fontsize=ann_size,
                    color="white" if v > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"Confusion matrix saved → {save_path}", "INFO")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _prepare_spl_run(
    spec: ProblemSpec, args: Any, device: torch.device, experiment_name: str
) -> dict:
    """Build what does not depend on lr, head width and batch size.

    Locates or compiles the SDD, loads the task data, builds the circuit and its
    MAP/marginal queries, the ground-truth consistency masks and, with
    ``--from-dimacs``, the relation of the formula to the ground truth. The
    circuit has no parameters of its own, so ``optuna`` reuses it across trials.
    """
    # ---- Locate SDD (from compile step or from a direct DIMACS file)
    if getattr(args, "from_dimacs", None) is not None:
        sdd_dir = args.output_dir_path / "sdd" / "from-dimacs" / spec.problem_id
        log(f"Compiling DIMACS from {args.from_dimacs} ...", "INFO")
        from fonesy.experiments.compile import compile_dimacs_file
        aux = getattr(args, "auxiliary_num_vars", 0) or 0
        result = compile_dimacs_file(Path(args.from_dimacs), sdd_dir, auxiliary_num_vars=aux)
        if result["status"] != "ok":
            raise RuntimeError(
                f"DIMACS compilation failed: {result.get('error_message', 'unknown error')}"
            )
        sdd_path   = sdd_dir / f"{spec.problem_id}.sdd"
        vtree_path = sdd_dir / f"{spec.problem_id}.vtree"
    else:
        _compile_name = getattr(args, "compile_experiment_name", None) or experiment_name
        sdd_dir    = args.output_dir_path / "sdd" / _compile_name
        sdd_path   = sdd_dir / f"{spec.problem_id}.sdd"
        vtree_path = sdd_dir / f"{spec.problem_id}.vtree"

        if not sdd_path.exists():
            raise FileNotFoundError(
                f"SDD not found: {sdd_path}\n"
                "Run the corresponding 'evaluate' + 'compile' commands first, "
                "or pass --from-dimacs <path> to compile a DIMACS file directly."
            )

    # ---- SPL visual task
    log(f"Loading SPL task '{spec.task_name}' ...", "INFO")
    spl_task = TaskFactory.get_task(spec.task_name, root=str(_DATA_DIR))

    # ---- Compile conditional circuit and create MAP/Integrate queries
    log(f"Compiling circuit from {sdd_path} ...", "INFO")
    circuit, gf_specs = _build_circuit(sdd_path, vtree_path)
    circuit = circuit.to(device)

    gf_size = _total_gate_param_size(gf_specs)
    log(
        f"Circuit has {circuit.num_variables} variables and {len(gf_specs)} gate "
        f"function(s) needing {gf_size} total params: "
        f"{ {k: tuple(v) for k, v in gf_specs.items()} }.",
        "INFO",
    )

    map_query       = _make_map_query(circuit)
    integrate_query = make_integrate_query(circuit)

    # Consistency is always measured against the ground-truth target.
    ground_truth_dimacs = _DATA_DIR / "auto-nesy-bench" / "targets" / f"{spec.problem_id}.dimacs"
    try:
        gt_masks = build_ground_truth_masks(ground_truth_dimacs, circuit.num_variables, device)
        log(f"Ground-truth consistency reference: {ground_truth_dimacs}", "INFO")
    except ValueError as exc:
        log(f"Skipping consistency metric: {exc}", "WARNING")
        gt_masks = None

    # REL (equal / permissive / too_strict / incomparable) and MC-R of the
    # --from-dimacs formula w.r.t. the ground truth.
    formula_rel: dict = {"formula_rel": "n/a", "mc_ratio": None}
    if getattr(args, "from_dimacs", None) is not None:
        formula_rel = formula_vs_oracle(
            args.from_dimacs, ground_truth_dimacs,
            getattr(args, "auxiliary_num_vars", 0) or 0,
        )
        log(
            f"LLM formula vs oracle: {formula_rel['formula_rel']} "
            f"(model-count ratio {formula_rel['mc_ratio']})",
            "INFO",
        )

    return {
        "sdd_path":        sdd_path,
        "spl_task":        spl_task,
        "circuit":         circuit,
        "gf_specs":        gf_specs,
        "gf_size":         gf_size,
        "map_query":       map_query,
        "integrate_query": integrate_query,
        "gt_masks":        gt_masks,
        "formula_rel":     formula_rel,
    }


def _fmt_metrics(metrics: dict) -> str:
    """One-line ``k=v`` rendering of a metrics dict."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v}" if isinstance(v, int) else f"{k}={v:.4f}")
    return " ".join(parts)


def _train_and_evaluate(
    prepared: dict,
    spec: ProblemSpec,
    args: Any,
    device: torch.device,
    epoch_callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Build a fresh backbone, head and optimizer, train and evaluate.

    ``epoch_callback(epoch, entry)`` is called after every validation (used by
    ``optuna`` for pruning).
    """
    spl_task        = prepared["spl_task"]
    circuit         = prepared["circuit"]
    gf_specs        = prepared["gf_specs"]
    gf_size         = prepared["gf_size"]
    map_query       = prepared["map_query"]
    integrate_query = prepared["integrate_query"]
    gt_masks        = prepared["gt_masks"]

    # ---- Backbone (from scratch, jointly trained) + MLP head
    backbone_name = args.backbone or spec.default_backbone
    trainable_params: list = []
    if backbone_name == "mlp":
        log(
            f"Building 'mlp' backbone from scratch ({args.mlp_layers} layers, "
            f"hidden={args.mlp_hidden}) over the precomputed {spl_task.feature_dim}-d features ...",
            "INFO",
        )
        backbone = NetworkFactory.get_network(
            backbone_name,
            n_classes=spl_task.n_classes,
            in_channels=spl_task.in_channels,
            in_dim=spl_task.feature_dim,
            hidden_dim=args.mlp_hidden,
            n_layers=args.mlp_layers,
        ).to(device)
        feature_dim = _get_feature_dim(backbone)
        trainable_params += list(backbone.features.parameters())
    elif backbone_name is not None:
        log(f"Building backbone '{backbone_name}' from scratch (n_classes={spl_task.n_classes}, in_channels={spl_task.in_channels}) ...", "INFO")
        backbone = NetworkFactory.get_network(
            backbone_name,
            n_classes=spl_task.n_classes,
            in_channels=spl_task.in_channels,
            image_size=spl_task.image_size,
        ).to(device)
        feature_dim = _get_feature_dim(backbone)
        trainable_params += list(backbone.features.parameters())
    else:
        backbone = None
        feature_dim = spl_task.feature_dim
        log(f"No backbone for '{args.spl_task}' — feeding precomputed {feature_dim}-d features to the head directly.", "INFO")

    log(f"MLP head: Linear({feature_dim}, {args.spl_hidden}) → ReLU → Linear({args.spl_hidden}, {gf_size})", "INFO")
    head = nn.Sequential(
        nn.Linear(feature_dim, args.spl_hidden),
        nn.ReLU(),
        nn.Linear(args.spl_hidden, gf_size),
    ).to(device)
    trainable_params += list(head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.spl_lr)

    # ---- Training loop
    train_loader = spl_task.train_loader(args.spl_batch_size)
    val_loader   = spl_task.val_loader(args.spl_batch_size)
    test_loader  = spl_task.test_loader(args.spl_batch_size)

    history: list[dict] = []
    for epoch in range(1, args.spl_epochs + 1):
        tr_loss = _train_epoch(
            backbone, head, circuit, gf_specs, spec, train_loader, optimizer, device,
        )
        val_metrics, _, _ = _eval_epoch(
            backbone, head, map_query, integrate_query, gf_specs, spec, val_loader, device,
            gt_masks=gt_masks,
        )
        entry = {"epoch": epoch, "train_loss": tr_loss}
        entry.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(entry)
        metrics_str = _fmt_metrics(val_metrics)
        log(f"Epoch {epoch}/{args.spl_epochs}: train_loss={tr_loss:.4f} {metrics_str}", "INFO")
        if epoch_callback is not None:
            epoch_callback(epoch, entry)

    test_metrics, test_true, test_pred = _eval_epoch(
        backbone, head, map_query, integrate_query, gf_specs, spec, test_loader, device,
        gt_masks=gt_masks,
    )
    log("Test: " + _fmt_metrics(test_metrics), "INFO")

    return {
        "backbone":      backbone,
        "head":          head,
        "backbone_name": backbone_name,
        "feature_dim":   feature_dim,
        "gf_size":       gf_size,
        "history":       history,
        "test_metrics":  test_metrics,
        "test_true":     test_true,
        "test_pred":     test_pred,
    }


def main(
    experiment_name: str,
    results_output: Any,
    stats_output: Any,
    args: Any,
    device: torch.device,
) -> None:
    """Train an SPL model (backbone + head, from scratch) and evaluate it."""
    spec = PROBLEM_SPECS[args.spl_task]

    prepared = _prepare_spl_run(spec, args, device, experiment_name)
    run = _train_and_evaluate(prepared, spec, args, device)

    sdd_path      = prepared["sdd_path"]
    backbone      = run["backbone"]
    head          = run["head"]
    backbone_name = run["backbone_name"]
    feature_dim   = run["feature_dim"]
    gf_size       = run["gf_size"]
    history       = run["history"]
    test_metrics  = run["test_metrics"]
    test_true     = run["test_true"]
    test_pred     = run["test_pred"]

    # ---- Save trained model
    save_dir   = args.output_dir_path / "splmodels" / args.spl_task
    model_path = save_dir / "model.pt"
    if not args.dry_run:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head":              head.state_dict(),
                "backbone_features": backbone.features.state_dict() if backbone is not None else None,
                "backbone_name":     backbone_name,
                "spl_task":          args.spl_task,
                "spl_hidden":        args.spl_hidden,
                "feature_dim":       feature_dim,
                "gf_size":           gf_size,
                # Checked when reloading the head.
                "gf_specs":          {k: list(v) for k, v in prepared["gf_specs"].items()},
            },
            model_path,
        )
        log(f"SPL model saved to {model_path}", "INFO")
    else:
        log(f"Dry-run: skipping model save to {model_path}", "INFO")

    # ---- Write results / stats JSON
    formula_rel = prepared.get("formula_rel") or {"formula_rel": "n/a", "mc_ratio": None}
    results = {
        "experiment": experiment_name,
        "spl_task":   args.spl_task,
        "backbone":   backbone_name,
        "sdd_path":   str(sdd_path),
        "model_path": str(model_path),
        "epochs":     args.spl_epochs,
        "formula_vs_oracle": formula_rel,
        "training":   history,
        "test":       test_metrics,
    }
    json.dump(results, results_output, indent=4)

    best  = max(history, key=lambda r: r["val_accuracy"]) if history else {}
    stats = {
        "best_val_accuracy": best.get("val_accuracy"),
        "best_val_f1":       best.get("val_f1"),
        "best_val_epoch":    best.get("epoch"),
        "formula_rel":       formula_rel["formula_rel"],
        "formula_mc_ratio":  formula_rel["mc_ratio"],
    }
    stats.update({f"final_test_{k}": v for k, v in test_metrics.items()})
    json.dump(stats, stats_output, indent=4)

    # ---- Confusion matrices (test split, skipped on dry-run)
    if not args.dry_run and test_true.size:
        plot_dir = args.output_dir_path / "splplots" / args.spl_task
        plot_dir.mkdir(parents=True, exist_ok=True)

        log("Saving test-set confusion matrix/matrices …", "INFO")
        if spec.output_kind == "multi_label" and len(spec.output_names) > 12:
            log(
                f"Skipping per-column confusion matrices for '{args.spl_task}' "
                f"({len(spec.output_names)} columns — too many small plots to be useful).",
                "INFO",
            )
        elif spec.output_kind == "multi_label":
            for j, name in enumerate(spec.output_names):
                _save_confusion_matrix_pdf(
                    test_true[:, j], test_pred[:, j],
                    title=f"{name} — {args.spl_task}",
                    labels=[0, 1],
                    save_path=plot_dir / f"{name}_cm.pdf",
                )
        else:
            _save_confusion_matrix_pdf(
                test_true, test_pred,
                title=args.spl_task,
                labels=list(range(spec.n_classes)),
                save_path=plot_dir / f"{args.spl_task}_cm.pdf",
            )
