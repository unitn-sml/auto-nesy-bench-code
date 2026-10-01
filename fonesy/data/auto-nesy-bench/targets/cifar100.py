"""
CIFAR-100 (``cifar100``) oracle (100 vars): exactly-one mutual-exclusion constraint
over 100 class indicators - the semi-supervised classification constraint
from Section 4 (MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100 all share it),
generalized from the paper's illustrative n=3 example
((x1 v x2 v x3) & (not x1 v not x2) & (not x1 v not x3) & (not x2 v not x3))
to n=100. Same constraint as fashion.py, just over 100 classes
instead of 10 (and, unlike fashion, not exhaustively enumerable).

One-hot encoding: variable i+1 (1-indexed DIMACS) is True iff the image
belongs to class i.

Total: 100 Boolean variables.
"""

N_VARS = 100

VARIABLE_NAMES = [f"class_{i}" for i in range(100)]

VARIABLE_DESCRIPTIONS = [f"the image belongs to class {i}" for i in range(100)]

# A valid assignment is one-hot: sample one directly.
def POSITIVE_SAMPLER(rng):
    """Return a random valid one-hot assignment over the 100 classes."""
    i = rng.randint(0, 99)
    return [1 if j == i else 0 for j in range(100)]


def check(assignment) -> bool:
    """
    assignment[i] = 0 or 1 for DIMACS variable i+1.
    Returns True iff exactly one of the 100 class indicators is True.
    """
    return sum(assignment) == 1
