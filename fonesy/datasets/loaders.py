from torch.utils.data import DataLoader


def _collate(batch):
    """Turn a list of ``(input, target, problem, id)`` tuples into four lists."""
    inputs, targets, problems, ids = zip(*batch)
    return list(inputs), list(targets), list(problems), list(ids)


def make_dataloader(dataset, batch_size=16, num_workers=2):
    """Deterministic (unshuffled) evaluation dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
