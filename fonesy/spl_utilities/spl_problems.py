"""Registry of the SPL tasks: how data labels map to CNF assignments.

Each :class:`ProblemSpec` is built from the task's oracle
(``fonesy/data/auto-nesy-bench/targets/<id>.py``) and defines how raw labels
are encoded as concept variables, how output variables are derived from them
(through the oracle's ``check`` for multi-label oracles, by arithmetic for the
one-hot MNIST tasks, and not at all for tasks whose whole assignment is
observed), and how output bits decode to a downstream prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

OutputKind = Literal["binary_radix", "onehot", "multi_label"]


from fonesy.utils.oracles import load_oracle as _load_oracle


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------


def _one_hot_block(values: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(values.long(), num_classes=size).float()


def _radix_decode(bits: torch.Tensor) -> torch.Tensor:
    """Decode a LSB-first binary block ``(B, n_bits)`` to an integer ``(B,)``."""
    n_bits = bits.shape[1]
    powers = torch.tensor(
        [2**i for i in range(n_bits)], dtype=torch.float32, device=bits.device
    )
    return (bits.float() * powers).sum(dim=1).long()


def _onehot_decode(bits: torch.Tensor) -> torch.Tensor:
    """Decode a one-hot (or near-one-hot, e.g. MAP-predicted) block via argmax."""
    return bits.float().argmax(dim=1).long()


def _make_range_blocks_concept_encoder(
    blocks: list[dict],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an encoder for ``VALIDITY_SPEC={"kind": "range_blocks", ...}``.

    ``raw`` is ``(B, len(blocks))`` integer-valued (e.g. two MNIST digits);
    each column is LSB-first binary-encoded into ``block["n_bits"]`` bits.
    """

    def fn(raw: torch.Tensor) -> torch.Tensor:
        parts = []
        for i, block in enumerate(blocks):
            n_bits = block["n_bits"]
            values = raw[:, i].long()
            bits = torch.stack([(values >> b) & 1 for b in range(n_bits)], dim=1)
            parts.append(bits.float())
        return torch.cat(parts, dim=1)

    return fn


def _make_onehot_concept_encoder(
    block_sizes: list[int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an encoder that one-hot-encodes each integer-valued raw column."""

    def fn(raw: torch.Tensor) -> torch.Tensor:
        parts = [_one_hot_block(raw[:, i], size) for i, size in enumerate(block_sizes)]
        return torch.cat(parts, dim=1)

    return fn


def _passthrough_concept_encoder(raw: torch.Tensor) -> torch.Tensor:
    """Concepts already arrive as a ``(B, n_concept_vars)`` 0/1 tensor."""
    return raw.float()


def _make_canonical_pattern_concept_encoder(
    patterns: dict[int, tuple[int, ...]], n_label: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an encoder for the ``cifar10`` oracle's whole-assignment layout.

    ``raw`` is ``(B, 1)`` — a merged 8-way class index. Each row expands to
    ``n_concept designed-concept bits`` (the class's canonical pattern) ++
    ``n_label`` one-hot label bits.
    """
    n_concept = len(next(iter(patterns.values())))
    pat = torch.zeros(len(patterns), n_concept, dtype=torch.float32)
    for k, bits in patterns.items():
        pat[k] = torch.tensor(bits, dtype=torch.float32)

    def fn(raw: torch.Tensor) -> torch.Tensor:
        idx = raw.view(-1).long()
        concept_bits = pat.to(idx.device)[idx]
        label_bits = _one_hot_block(idx, n_label)
        return torch.cat([concept_bits, label_bits], dim=1)

    return fn


def _no_output(concept_bits: torch.Tensor) -> torch.Tensor:
    """For whole-assignment validity oracles: nothing is 'derived' beyond the
    concepts themselves (``encode_concepts`` already builds the full
    assignment) — returns an empty ``(B, 0)`` tensor."""
    return torch.zeros(concept_bits.shape[0], 0, dtype=torch.float32, device=concept_bits.device)


def _make_oracle_check_output(
    oracle, n_output_vars: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an output-deriver that dispatches through a multi-label ``oracle.check()``.

    ``check()`` is a plain-Python function over one sample's concept list, so
    this loops over the batch — batches are small relative to the
    circuit-inference cost that dominates the training/eval loop, so this is
    not a bottleneck.
    """

    def fn(concept_bits: torch.Tensor) -> torch.Tensor:
        batch = concept_bits.detach().to("cpu").round().long().tolist()
        out = torch.empty(len(batch), n_output_vars, dtype=torch.float32)
        for i, sample in enumerate(batch):
            result = oracle.check(sample)
            out[i] = torch.tensor([float(bool(x)) for x in result], dtype=torch.float32)
        return out.to(concept_bits.device)

    return fn


def _make_onehot_arithmetic_output(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    concept_block_sizes: list[int],
    output_block_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an output-deriver for the single-label "normal" one-hot oracles.

    These oracles only expose ``check(full_assignment) -> bool`` (validity),
    not a concepts->output function, so the output is computed directly via
    the known arithmetic op (``+``/``*``) the oracle's own ``POSITIVE_SAMPLER``
    uses, then one-hot encoded.
    """
    offsets = [0]
    for size in concept_block_sizes:
        offsets.append(offsets[-1] + size)

    def fn(concept_bits: torch.Tensor) -> torch.Tensor:
        d1 = concept_bits[:, offsets[0] : offsets[1]].argmax(dim=1)
        d2 = concept_bits[:, offsets[1] : offsets[2]].argmax(dim=1)
        result = op(d1, d2)
        return _one_hot_block(result, output_block_size)

    return fn


# ---------------------------------------------------------------------------
# ProblemSpec
# ---------------------------------------------------------------------------


@dataclass
class ProblemSpec:
    spl_task: str
    problem_id: str
    task_name: str
    default_backbone: str | None
    output_kind: OutputKind
    n_concept_vars: int
    n_output_vars: int
    n_classes: int | None  # multiclass problems only; None for multi_label
    output_names: list[str]
    _encode_concepts: Callable[[torch.Tensor], torch.Tensor]
    _derive_output: Callable[[torch.Tensor], torch.Tensor]
    # First variable scored at evaluation: n_concept_vars (only the outputs) by
    # default, 0 for tasks whose whole assignment is the prediction.
    eval_start_var: int = -1

    def __post_init__(self) -> None:
        if self.eval_start_var == -1:
            self.eval_start_var = self.n_concept_vars

    @property
    def n_vars(self) -> int:
        return self.n_concept_vars + self.n_output_vars

    def encode_concepts(self, raw_concepts: torch.Tensor) -> torch.Tensor:
        """Raw per-sample labels -> ``(B, n_concept_vars)`` float 0/1 tensor."""
        return self._encode_concepts(raw_concepts)

    def derive_output(self, concept_bits: torch.Tensor) -> torch.Tensor:
        """Concept bits -> ``(B, n_output_vars)`` float 0/1 tensor."""
        return self._derive_output(concept_bits)

    def encode_x_truth(self, raw_concepts: torch.Tensor) -> torch.Tensor:
        """Full circuit assignment ``(B, n_vars)`` = concept bits ++ derived output bits."""
        concept_bits = self.encode_concepts(raw_concepts)
        output_bits = self.derive_output(concept_bits)
        return torch.cat([concept_bits, output_bits], dim=1)

    def decode_downstream(self, output_bits: torch.Tensor) -> torch.Tensor:
        """Output bits -> downstream prediction.

        ``binary_radix``/``onehot``: ``(B,)`` long class index.
        ``multi_label``: unchanged ``(B, n_output_vars)`` per-action bits.
        """
        if self.output_kind == "binary_radix":
            return _radix_decode(output_bits)
        if self.output_kind == "onehot":
            return _onehot_decode(output_bits)
        return output_bits

    def class_to_output_bits(self, k: int) -> torch.Tensor:
        """The exact output-bit pattern for class ``k`` (multiclass problems only)."""
        if self.output_kind == "binary_radix":
            return torch.tensor(
                [(k >> b) & 1 for b in range(self.n_output_vars)], dtype=torch.float32
            )
        if self.output_kind == "onehot":
            v = torch.zeros(self.n_output_vars, dtype=torch.float32)
            v[k] = 1.0
            return v
        raise ValueError("class_to_output_bits is undefined for multi_label problems")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, ProblemSpec]:
    specs: dict[str, ProblemSpec] = {}

    # -- mn-add-bin (binary radix) ---------------------------------------------
    oracle = _load_oracle("mn-add-bin")
    blocks = oracle.VALIDITY_SPEC["blocks"]
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    assert n_concept + n_output == oracle.N_VARS
    specs["mn-add-bin"] = ProblemSpec(
        spl_task="mn-add-bin",
        problem_id="mn-add-bin",
        task_name="mnist-addition",
        default_backbone="conv4",
        output_kind="binary_radix",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=19,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_make_range_blocks_concept_encoder(blocks),
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    # -- mn-add (one-hot) --------------------------------------------------------
    oracle = _load_oracle("mn-add")
    block_layout = oracle.VALIDITY_SPEC["block_layout"]  # [10, 10, 19]
    concept_blocks, output_block = block_layout[:-1], block_layout[-1]
    n_concept = sum(concept_blocks)
    assert n_concept + output_block == oracle.N_VARS
    specs["mn-add"] = ProblemSpec(
        spl_task="mn-add",
        problem_id="mn-add",
        task_name="mnist-addition",
        default_backbone="conv4",
        output_kind="onehot",
        n_concept_vars=n_concept,
        n_output_vars=output_block,
        n_classes=output_block,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_make_onehot_concept_encoder(concept_blocks),
        _derive_output=_make_onehot_arithmetic_output(torch.add, concept_blocks, output_block),
    )

    # -- mn-mul-bin (binary radix) ---------------------------------------------
    oracle = _load_oracle("mn-mul-bin")
    blocks = oracle.VALIDITY_SPEC["blocks"]
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    specs["mn-mul-bin"] = ProblemSpec(
        spl_task="mn-mul-bin",
        problem_id="mn-mul-bin",
        task_name="mnist-multiplication",
        default_backbone="conv4",
        output_kind="binary_radix",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=82,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_make_range_blocks_concept_encoder(blocks),
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    # -- mn-mul (one-hot) --------------------------------------------------------
    oracle = _load_oracle("mn-mul")
    block_layout = oracle.VALIDITY_SPEC["block_layout"]  # [10, 10, 82]
    concept_blocks, output_block = block_layout[:-1], block_layout[-1]
    n_concept = sum(concept_blocks)
    assert n_concept + output_block == oracle.N_VARS
    specs["mn-mul"] = ProblemSpec(
        spl_task="mn-mul",
        problem_id="mn-mul",
        task_name="mnist-multiplication",
        default_backbone="conv4",
        output_kind="onehot",
        n_concept_vars=n_concept,
        n_output_vars=output_block,
        n_classes=output_block,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_make_onehot_concept_encoder(concept_blocks),
        _derive_output=_make_onehot_arithmetic_output(torch.mul, concept_blocks, output_block),
    )

    # -- bdd-oia (multi-label) ------------------------------------------------
    oracle = _load_oracle("bdd-oia")
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    specs["bdd-oia"] = ProblemSpec(
        spl_task="bdd-oia",
        problem_id="bdd-oia",
        task_name="bdd-oia",
        default_backbone="mlp",
        output_kind="multi_label",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=None,
        output_names=oracle.ACTION_NAMES,
        _encode_concepts=_passthrough_concept_encoder,
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    # -- bdd-oia-2 (multi-label) -----------------------------------------------
    oracle = _load_oracle("bdd-oia-2")
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    specs["bdd-oia-2"] = ProblemSpec(
        spl_task="bdd-oia-2",
        problem_id="bdd-oia-2",
        task_name="bdd-oia-simple",
        default_backbone="mlp",
        output_kind="multi_label",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=None,
        output_names=oracle.ACTION_NAMES,
        _encode_concepts=_passthrough_concept_encoder,
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    # -- sudoku (whole-assignment validity, 16 one-hot blocks of 4) -----------
    oracle = _load_oracle("sudoku")
    blocks = oracle.VALIDITY_SPEC["block_layout"]
    specs["sudoku"] = ProblemSpec(
        spl_task="sudoku",
        problem_id="sudoku",
        task_name="sudoku",
        default_backbone="conv4",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder(blocks),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- cle4evr (whole-assignment validity, 4 one-hot blocks of 3) -----------
    oracle = _load_oracle("cle4evr")
    blocks = oracle.VALIDITY_SPEC["block_layout"]
    specs["cle4evr"] = ProblemSpec(
        spl_task="cle4evr",
        problem_id="cle4evr",
        task_name="cle4evr",
        default_backbone="fasterrcnn",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder(blocks),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- road-r (whole-assignment validity, no VALIDITY_SPEC — mixed groups) --
    oracle = _load_oracle("road-r")
    specs["road-r"] = ProblemSpec(
        spl_task="road-r",
        problem_id="road-r",
        task_name="road-r",
        default_backbone="slowfast",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_passthrough_concept_encoder,
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- kand-logic (whole-assignment validity, 12 one-hot blocks of 3) -------
    oracle = _load_oracle("kand-logic")
    blocks = oracle.VALIDITY_SPEC["block_layout"]
    specs["kand-logic"] = ProblemSpec(
        spl_task="kand-logic",
        problem_id="kand-logic",
        task_name="kand-logic",
        default_backbone="fasterrcnn",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder(blocks),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- kand-logic-2 (whole-assignment validity, 8 one-hot blocks of 3) --------
    oracle = _load_oracle("kand-logic-2")
    blocks = oracle.VALIDITY_SPEC["block_layout"]
    specs["kand-logic-2"] = ProblemSpec(
        spl_task="kand-logic-2",
        problem_id="kand-logic-2",
        task_name="kand-logic-simple",
        default_backbone="fasterrcnn",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder(blocks),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- sushi (whole-assignment validity, 4 one-hot blocks of 4) -------------
    # The 6-item context ranking is only the network input; the CNF covers the
    # 4x4 permutation matrix of the 4-item label ranking.
    oracle = _load_oracle("sushi")
    blocks = oracle.VALIDITY_SPEC["block_layout"]
    specs["sushi"] = ProblemSpec(
        spl_task="sushi",
        problem_id="sushi",
        task_name="sushi",
        default_backbone="mlp",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder(blocks),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- fashion (whole-assignment validity, 1 one-hot block of 10) ------------
    oracle = _load_oracle("fashion")
    specs["fashion"] = ProblemSpec(
        spl_task="fashion",
        problem_id="fashion",
        task_name="fashion-mut-ex",
        default_backbone="conv4",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder([oracle.N_VARS]),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- cifar100 (whole-assignment validity, 1 one-hot block of 100) ----------
    oracle = _load_oracle("cifar100")
    specs["cifar100"] = ProblemSpec(
        spl_task="cifar100",
        problem_id="cifar100",
        task_name="cifar100-mut-ex",
        default_backbone="resnet18",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_onehot_concept_encoder([oracle.N_VARS]),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- cifar10 (whole-assignment validity: 7 designed concepts + 8-way label)
    oracle = _load_oracle("cifar10")
    _CIFAR10_N_LABEL = 8
    specs["cifar10"] = ProblemSpec(
        spl_task="cifar10",
        problem_id="cifar10",
        task_name="cifar10",
        default_backbone="resnet18",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_make_canonical_pattern_concept_encoder(
            oracle._CANONICAL_PATTERNS, _CIFAR10_N_LABEL
        ),
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- warcraft (whole-assignment validity, 24 edge vars) --------------------
    oracle = _load_oracle("warcraft")
    specs["warcraft"] = ProblemSpec(
        spl_task="warcraft",
        problem_id="warcraft",
        task_name="warcraft-path",
        default_backbone="conv4",
        output_kind="multi_label",
        n_concept_vars=oracle.N_VARS,
        n_output_vars=0,
        n_classes=None,
        output_names=oracle.VARIABLE_NAMES,
        _encode_concepts=_passthrough_concept_encoder,
        _derive_output=_no_output,
        eval_start_var=0,
    )

    # -- chx (multi-label oracle: 4 findings -> 5-way one-hot severity) ------
    oracle = _load_oracle("chx")
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    specs["chx"] = ProblemSpec(
        spl_task="chx",
        problem_id="chx",
        task_name="chx",
        default_backbone="resnet18",
        output_kind="onehot",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=n_output,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_passthrough_concept_encoder,
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    # -- cebab (multi-label oracle: 5 one-hot concepts -> 5-way one-hot label)
    oracle = _load_oracle("cebab")
    n_concept, n_output = oracle.N_CONCEPT_VARS, oracle.N_VARS - oracle.N_CONCEPT_VARS
    specs["cebab"] = ProblemSpec(
        spl_task="cebab",
        problem_id="cebab",
        task_name="cebab",
        default_backbone="bert",
        output_kind="onehot",
        n_concept_vars=n_concept,
        n_output_vars=n_output,
        n_classes=n_output,
        output_names=oracle.VARIABLE_NAMES[n_concept:],
        _encode_concepts=_make_onehot_concept_encoder([4, 4, 4, 4, 4]),
        _derive_output=_make_oracle_check_output(oracle, n_output),
    )

    for spec in specs.values():
        assert spec.n_vars == _load_oracle(spec.problem_id).N_VARS, (
            f"{spec.spl_task}: n_vars ({spec.n_vars}) does not match oracle "
            f"N_VARS ({_load_oracle(spec.problem_id).N_VARS})"
        )

    return specs


PROBLEM_SPECS: dict[str, ProblemSpec] = _build_registry()
