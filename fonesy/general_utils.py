"""Global utility functions shared across the fonesy package."""

import argparse
import hashlib
import logging
import random
import torch

import numpy as np

LOG_LEVEL = logging.INFO


def set_log_level(level_str: str) -> None:
    """Set the global log level used by :func:`log`.

    Args:
        level_str: Level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    """
    global LOG_LEVEL
    LOG_LEVEL = getattr(logging, level_str.upper(), logging.INFO)


def log(message: str, level_str: str = "INFO") -> None:
    """Log *message* if its level is at or above the global :data:`LOG_LEVEL`.

    Args:
        message: Text to log.
        level_str: Severity of this message (default ``"INFO"``).
    """
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    if numeric_level >= LOG_LEVEL:
        logging.log(numeric_level, message)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seeding CUDA before vLLM starts its workers would initialize CUDA early.
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_basename(args: argparse.Namespace, separator: str = "_") -> str:
    """Experiment name built from the values of all CLI arguments.

    ``/`` in values (e.g. HuggingFace model ids) becomes ``--``; list and dict
    values are replaced by a short hash, and names longer than 150 characters
    are truncated and suffixed with a hash, to keep file names valid.
    """
    args_str = []
    for key, value in vars(args).items():
        if key in ("func", "output_dir_path"):
            continue
        if isinstance(value, dict):
            safe_value = hashlib.sha1(repr(sorted(value.items())).encode()).hexdigest()[:10]
        elif isinstance(value, (list, tuple)):
            safe_value = hashlib.sha1(repr(list(value)).encode()).hexdigest()[:10]
        else:
            safe_value = str(value).replace("/", "--").replace("\\", "--")
        args_str.append(safe_value)
    experiment_name = f"experiment_{separator.join(args_str)}"
    max_len = 150
    if len(experiment_name) > max_len:
        digest = hashlib.sha1(experiment_name.encode()).hexdigest()[:10]
        experiment_name = f"{experiment_name[:max_len]}_{digest}"
    return experiment_name


def set_device(device_str: str) -> torch.device:
    """Return the appropriate :class:`torch.device` for *device_str*.

    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        device_str: ``"cuda"`` or ``"cpu"``.

    Returns:
        A :class:`torch.device` object.
    """
    # No torch.cuda.is_available() check: it would initialize CUDA before vLLM.
    return torch.device(device_str)


def load_file(path: str) -> str:
    """Read a text file and return its contents stripped of leading/trailing whitespace.

    Args:
        path: Path to the file.

    Returns:
        The file contents as a string.
    """
    with open(path, "r") as f:
        return f.read().strip()
