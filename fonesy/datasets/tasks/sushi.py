"""SUSHI (``sushi``): 6-item preference ranking -> ranking of 4 other items.

Following Xu et al. (2018), each respondent's ranking of 10 sushi items
(SUSHI3, ``sushi3a.5000.10.order``) is split into a 6-item context
(``DATA_IND_0``, fed to an MLP as a 36-d one-hot permutation matrix) and a
4-item label (``LABEL_IND_0``, 4 relative ranks forming the 4x4 permutation
matrix of the CNF). SUSHI3 cannot be redistributed: ``make sushi`` downloads it
from Kamishima's archive into ``fonesy/data/sushi/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fonesy.datasets.tasks.base import Task

N_CONTEXT_ITEMS = 6
N_LABEL_ITEMS = 4
DATA_IND_0 = [0, 1, 2, 4, 6, 7]  # 6-item context, 0-indexed item ids
LABEL_IND_0 = [3, 5, 8, 9]  # 4-item label, 0-indexed item ids

_N_TRAIN = 4000
_N_VAL = 500
_N_TEST = 500

_ORDER_PATH = Path("sushi") / "raw" / "sushi3-2016" / "sushi3a.5000.10.order"


def _relative_ranks(order: list[int], item_ids: list[int]) -> list[int]:
    """Given a full preference ``order`` (position -> item id, most to least
    preferred) and a subset ``item_ids``, return each item's *relative* rank
    (0 = most preferred) within just that subset, in ``item_ids`` order."""
    item_to_global_rank = {item: pos for pos, item in enumerate(order)}
    global_ranks = [item_to_global_rank[i] for i in item_ids]
    order_within_subset = sorted(range(len(item_ids)), key=lambda k: global_ranks[k])
    relative = [0] * len(item_ids)
    for rel_rank, idx in enumerate(order_within_subset):
        relative[idx] = rel_rank
    return relative


class _SushiSplit(Dataset):
    def __init__(self, orders: list[list[int]]) -> None:
        self._orders = orders

    def __len__(self) -> int:
        return len(self._orders)

    def __getitem__(self, idx: int):
        order = self._orders[idx]
        context_ranks = _relative_ranks(order, DATA_IND_0)
        label_ranks = _relative_ranks(order, LABEL_IND_0)

        context_onehot = torch.zeros(N_CONTEXT_ITEMS * N_CONTEXT_ITEMS)
        for i, r in enumerate(context_ranks):
            context_onehot[i * N_CONTEXT_ITEMS + r] = 1.0

        label = torch.tensor(label_ranks, dtype=torch.long)
        return context_onehot, label


def parse_order_file(text: str) -> list[list[int]]:
    """Parse a Kamishima ``sushi3*.order`` file's body into a list of
    10-item preference orders (position -> item id, most to least preferred).

    Each non-header line is ``"0 10 <item0> ... <item9>"`` — the two leading
    fields are a fixed dummy-id/count pair, not part of the ranking.
    """
    lines = text.strip().splitlines()[1:]  # skip the "<n> <n_items>" header
    return [[int(x) for x in line.split()[2:]] for line in lines]


class SushiTask(Task):
    """Sushi preference-ranking permutation task (Xu et al. 2018).

    No CNN backbone — loaders yield ``(context_onehot[36], label_ranks[4])``
    fed directly to the SPL head via an MLP backbone. ``n_classes``/
    ``in_channels``/``image_size`` are unused placeholders (satisfying the
    ``Task`` contract); use ``feature_dim`` to size the head's input layer.
    """

    n_classes: int = 2  # unused — no CNN/classifier for this task
    in_channels: int = 1  # unused
    image_size: int = 1  # unused
    feature_dim: int = N_CONTEXT_ITEMS * N_CONTEXT_ITEMS  # 36

    def __init__(self, root: str, seed: int = 0) -> None:
        order_path = Path(root) / _ORDER_PATH
        if not order_path.exists():
            raise FileNotFoundError(
                f"SUSHI3 preference data not found at '{order_path}'.\n"
                "Run 'make sushi'."
            )
        orders = parse_order_file(order_path.read_text())
        if len(orders) != _N_TRAIN + _N_VAL + _N_TEST:
            raise ValueError(
                f"Expected {_N_TRAIN + _N_VAL + _N_TEST} respondents in "
                f"'{order_path}', found {len(orders)}."
            )

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(orders), generator=rng).tolist()
        orders = [orders[i] for i in perm]

        self._train = orders[:_N_TRAIN]
        self._val = orders[_N_TRAIN : _N_TRAIN + _N_VAL]
        self._test = orders[_N_TRAIN + _N_VAL :]

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(_SushiSplit(self._train), batch_size=batch_size, shuffle=True)

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(_SushiSplit(self._val), batch_size=batch_size, shuffle=False)

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(_SushiSplit(self._test), batch_size=batch_size, shuffle=False)
