#!/usr/bin/env python3
r"""Generate the Kand-Logic data (``kand-logic`` and ``kand-logic-2``).

Single-figure Kandinsky images are rendered with rsbench's
``SyntheticKandinksyGenerator`` (``--rsbench-path``, a clone of
https://github.com/unitn-sml/rsbench-code). Pairs of images sharing a
shape or color pattern (same / pair / diff) are combined into samples, and
every sample is checked against the task oracle before being written.
rsbench's shape/color codes are remapped to the oracle's variable order.

Usage::

    python scripts/generate_kand_logic.py --rsbench-path external/rsbench-code \
        --variant kand-logic --output-dir fonesy/data/kand-logic
    python scripts/generate_kand_logic.py --rsbench-path external/rsbench-code \
        --variant kand-logic-2 --output-dir fonesy/data/kand-logic-2
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# rsbench-code's internal shape/color integer codes vs. our oracle's
# VARIABLE_NAMES order:
#   rsbench shape: square=4, circle=5, triangle=6
#   oracle  shape: square=0, triangle=1, circle=2   (see VARIABLE_NAMES)
#   rsbench color: red=1, yellow=2, blue=3
#   oracle  color: red=0, blue=1, yellow=2
_RSBENCH_SHAPE_TO_LOCAL = {4: 0, 6: 1, 5: 2}
_RSBENCH_COLOR_TO_LOCAL = {1: 0, 3: 1, 2: 2}

_VARIANTS = {
    "kand-logic": {"n_shapes": 3, "oracle_module": "kand-logic", "has_pair": True},
    "kand-logic-2": {"n_shapes": 2, "oracle_module": "kand-logic-2", "has_pair": False},
}


def _build_generator(rsbench_rssgen_path: Path, n_shapes: int):
    """Instantiate SyntheticKandinksyGenerator directly (bypassing the CLI/YAML
    layer, which hard-rejects n_figures=1) with a placeholder logic formula —
    never actually used for filtering, since generate_dataset() (the only
    consumer of self.logic/self.aggregator_logic for accept/reject decisions)
    is never called; only generate_synthetic_data() is, which only uses them
    to compute a (here, discarded) label.
    """
    sys.path.insert(0, str(rsbench_rssgen_path))
    from rssgen.generators.kandinksy_generator import SyntheticKandinksyGenerator
    from rssgen.generators.utils import get_exp

    # rsbench calls random.randint with float bounds, rejected by recent Python.
    _orig_randint = random.randint
    random.randint = lambda a, b: _orig_randint(int(a), int(b))

    symbols = [f"s{i}" for i in range(2 * n_shapes)]
    # Placeholder rule (its value is unused); "True" would not support .subs().
    placeholder_logic = get_exp(symbols, f"{symbols[0]} >= 0")
    aggregator_symbols = ["p1"]
    aggregator_logic = get_exp(aggregator_symbols, "p1")

    return SyntheticKandinksyGenerator(
        output_path="/tmp/unused_rssgen_output_path",  # never written to — only generate_dataset() uses this
        val_prop=0.0,
        test_prop=0.0,
        ood_prop=0.0,
        shapes=["square", "circle", "triangle"],
        colors=["red", "yellow", "blue"],
        logic=placeholder_logic,
        n_shapes=n_shapes,
        n_figures=1,
        symbols=symbols,
        aggregator_logic=aggregator_logic,
        sample_size=1,
        aggregator_symbols=aggregator_symbols,
    )


def _generate_pool(generator, n_images: int, n_shapes: int, rng: random.Random):
    """Returns [(PIL image, local_concepts[2*n_shapes]), ...] — one independent,
    unfiltered single-figure image per entry, concepts remapped to local order.
    """
    pool = []
    for _ in range(n_images):
        synthetic_image, _label, meta = generator.generate_synthetic_data()
        image = synthetic_image["image"]
        raw_concepts = meta["concepts"][0]  # n_figures=1 -> exactly one figure's concepts
        local_concepts = []
        for i in range(0, len(raw_concepts), 2):
            local_concepts.append(_RSBENCH_SHAPE_TO_LOCAL[raw_concepts[i]])
            local_concepts.append(_RSBENCH_COLOR_TO_LOCAL[raw_concepts[i + 1]])
        pool.append((image, local_concepts))
    return pool


def _categorize(local_concepts: list[int], n_shapes: int, same, diff, pair):
    """(color_category, shape_category) for one image, using the oracle's own predicates."""
    shapes = [local_concepts[2 * p] for p in range(n_shapes)]
    colors = [local_concepts[2 * p + 1] for p in range(n_shapes)]

    def _cat(lst):
        if same(lst):
            return "same"
        if diff(lst):
            return "diff"
        if pair is not None and pair(lst):
            return "pair"
        raise AssertionError(f"list {lst} matched no category — oracle predicates not exhaustive?")

    return _cat(colors), _cat(shapes)


def _build_pairs(pool, n_shapes, same, diff, pair, n_pairs_needed, rng):
    """Pair images sharing at least one matching category (color or shape) —
    guarantees oracle.check() validity by construction (its condition is an
    OR over color-category-match / shape-category-match)."""
    color_buckets: dict[str, list[int]] = {}
    shape_buckets: dict[str, list[int]] = {}
    categories = []
    for idx, (_, concepts) in enumerate(pool):
        c_cat, s_cat = _categorize(concepts, n_shapes, same, diff, pair)
        categories.append((c_cat, s_cat))
        color_buckets.setdefault(c_cat, []).append(idx)
        shape_buckets.setdefault(s_cat, []).append(idx)

    pairs: list[tuple[int, int]] = []
    # Alternate between color-matched and shape-matched pairs for diversity.
    bucket_cycle = [color_buckets, shape_buckets]
    bucket_i = 0
    guard = 0
    while len(pairs) < n_pairs_needed and guard < n_pairs_needed * 50:
        guard += 1
        buckets = bucket_cycle[bucket_i % 2]
        bucket_i += 1
        non_trivial = [b for b in buckets.values() if len(b) >= 2]
        if not non_trivial:
            continue
        bucket = rng.choice(non_trivial)
        i, j = rng.sample(bucket, 2)
        pairs.append((i, j))
    if len(pairs) < n_pairs_needed:
        raise RuntimeError(
            f"Could only form {len(pairs)}/{n_pairs_needed} pairs — pool too small "
            f"(try a larger --pool-size)."
        )
    return pairs


def _verify_pair(local_a, local_b, oracle, n_shapes) -> bool:
    """Build the full assignment (one-hot per block) and check it against oracle.check()."""
    assignment = []
    for concept in local_a + local_b:
        block = [0, 0, 0]
        block[concept] = 1
        assignment.extend(block)
    return oracle.check(assignment)


def generate(
    rsbench_path: Path,
    variant: str,
    output_dir: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    pool_size: int,
    seed: int,
) -> None:
    cfg = _VARIANTS[variant]
    n_shapes = cfg["n_shapes"]
    from fonesy.utils.oracles import load_oracle

    oracle = load_oracle(cfg["oracle_module"])
    same, diff = oracle._same, oracle._diff
    pair = oracle._pair if cfg["has_pair"] else None

    rng = random.Random(seed)
    random.seed(seed)  # rsbench's generator uses the global `random` module internally

    generator = _build_generator(rsbench_path / "rssgen", n_shapes)
    print(f"Generating a pool of {pool_size} independent single-figure images...")
    pool = _generate_pool(generator, pool_size, n_shapes, rng)

    n_total = n_train + n_val + n_test
    print(f"Pairing {n_total} samples ({n_train} train / {n_val} val / {n_test} test)...")
    pairs = _build_pairs(pool, n_shapes, same, diff, pair, n_total, rng)
    rng.shuffle(pairs)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val : n_train + n_val + n_test],
    }

    n_verified = 0
    for split_name, split_pairs in splits.items():
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels: dict[str, list[int]] = {}

        for k, (i, j) in enumerate(split_pairs):
            image_a, concepts_a = pool[i]
            image_b, concepts_b = pool[j]

            assert _verify_pair(concepts_a, concepts_b, oracle, n_shapes), (
                f"Generated pair {k} in split '{split_name}' failed oracle.check() — "
                "pairing logic is broken, aborting before writing any more data."
            )
            n_verified += 1

            name_a, name_b = f"{k:05d}_A.png", f"{k:05d}_B.png"
            image_a.convert("RGB").save(images_dir / name_a)
            image_b.convert("RGB").save(images_dir / name_b)
            labels[f"{k:05d}"] = {"image_a": name_a, "image_b": name_b, "concepts": concepts_a + concepts_b}

        (split_dir / "labels.json").write_text(json.dumps(labels, indent=2))
        print(f"  {split_name}: {len(split_pairs)} samples written to {split_dir}")

    print(f"Verified {n_verified}/{n_total} samples satisfy {cfg['oracle_module']}.check() — all valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rsbench-path", type=Path, required=True, help="Path to a local clone of unitn-sml/rsbench-code")
    parser.add_argument("--variant", choices=sorted(_VARIANTS.keys()), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=4000, help="Single-figure image pool size to pair from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        rsbench_path=args.rsbench_path,
        variant=args.variant,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        pool_size=args.pool_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
