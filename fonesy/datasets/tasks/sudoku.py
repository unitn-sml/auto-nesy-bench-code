"""4x4 visual Sudoku (``sudoku``): a grid of MNIST digits 1-4.

Grids are drawn from all 288 valid 4x4 boards with random MNIST crops and
rendered as one ``(1, 112, 112)`` image; labels are the 16 cell values (0-3).
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fonesy.datasets.tasks.base import Task

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

_VAL_SPLIT = 0.1
_N_TRAIN = 4000
_N_VAL = 500
_N_TEST = 500


def _enumerate_all_boards() -> list[list[list[int]]]:
    """Enumerate every valid 4x4 Sudoku board (exactly 288) via backtracking."""
    boards: list[list[list[int]]] = []
    grid = [[0] * 4 for _ in range(4)]

    def _valid(r: int, c: int, v: int) -> bool:
        if v in grid[r]:
            return False
        if any(grid[rr][c] == v for rr in range(4)):
            return False
        br, bc = (r // 2) * 2, (c // 2) * 2
        for rr in range(br, br + 2):
            for cc in range(bc, bc + 2):
                if grid[rr][cc] == v:
                    return False
        return True

    def _backtrack(pos: int) -> None:
        if pos == 16:
            boards.append([row[:] for row in grid])
            return
        r, c = divmod(pos, 4)
        for v in range(1, 5):
            if _valid(r, c, v):
                grid[r][c] = v
                _backtrack(pos + 1)
                grid[r][c] = 0

    _backtrack(0)
    return boards


_ALL_BOARDS = _enumerate_all_boards()
assert len(_ALL_BOARDS) == 288, f"expected 288 valid 4x4 boards, got {len(_ALL_BOARDS)}"


def _images_by_digit(base_ds, digits=(1, 2, 3, 4)) -> dict[int, list[torch.Tensor]]:
    """Group a torchvision MNIST dataset's images by digit label, digits 1-4 only."""
    by_digit: dict[int, list[torch.Tensor]] = {d: [] for d in digits}
    for img, label in base_ds:
        if label in by_digit:
            by_digit[label].append(img)
    return by_digit


class SudokuDataset(Dataset):
    """Yields ``(grid_image, cell_values)`` pairs — a tiled 4x4 MNIST Sudoku board.

    Parameters
    ----------
    boards: list of valid 4x4 boards (values 1-4) to sample from.
    digit_images: dict {1: [...], 2: [...], 3: [...], 4: [...]} of ``(1,28,28)``
        MNIST crops for each digit, drawn from the appropriate MNIST split.
    n_samples: fixed dataset length — each sample independently draws a random
        board and, per cell, a random MNIST crop of the required digit.
    seed: seeds the board/crop selection for reproducibility.
    """

    def __init__(
        self,
        boards: list[list[list[int]]],
        digit_images: dict[int, list[torch.Tensor]],
        n_samples: int,
        seed: int,
    ) -> None:
        self.boards = boards
        self.digit_images = digit_images
        rng = random.Random(seed)
        self._samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(n_samples):
            board = rng.choice(boards)
            rows = []
            cell_values = []
            for i in range(4):
                row_imgs = [rng.choice(digit_images[board[i][j]]) for j in range(4)]
                rows.append(torch.cat(row_imgs, dim=2))  # (1, 28, 112)
                cell_values.extend(board[i][j] - 1 for j in range(4))
            grid_img = torch.cat(rows, dim=1)  # (1, 112, 112)
            self._samples.append((grid_img, torch.tensor(cell_values, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


class SudokuTask(Task):
    """4x4 visual Sudoku: MNIST-digit grid images, 64 one-hot cell-value bits.

    Loaders yield ``(grid_image, cell_values[16])`` — ``cell_values`` are
    0-indexed (0-3) per cell, matching the oracle's ``block_layout=[4]*16``.
    """

    n_classes: int = 4    # unused directly by SPL (no separate classifier head)
    in_channels: int = 1
    image_size: int = 112

    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ])
        full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_base = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        val_size = int(_VAL_SPLIT * len(full_train))
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(0)
        train_split, val_split = random_split(
            range(len(full_train)), [train_size, val_size], generator=generator
        )

        train_digit_images = _images_by_digit(Subset(full_train, list(train_split)))
        val_digit_images = _images_by_digit(Subset(full_train, list(val_split)))
        test_digit_images = _images_by_digit(test_base)

        self._train_ds = SudokuDataset(_ALL_BOARDS, train_digit_images, _N_TRAIN, seed=0)
        self._val_ds = SudokuDataset(_ALL_BOARDS, val_digit_images, _N_VAL, seed=1)
        self._test_ds = SudokuDataset(_ALL_BOARDS, test_digit_images, _N_TEST, seed=2)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
