"""One-hot helpers shared by the auto-nesy-bench oracles."""


def _one_hot_to_int(assignment, offset, size):
    """Decode a one-hot block; return -1 if not exactly one bit is set."""
    block = assignment[offset: offset + size]
    if sum(block) != 1:
        return -1
    return block.index(1)


def _int_to_one_hot(value, size):
    """Encode an int as a one-hot block of the given size."""
    block = [0] * size
    block[value] = 1
    return block
