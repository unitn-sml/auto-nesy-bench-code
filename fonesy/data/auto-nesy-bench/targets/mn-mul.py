"""
MNIST-Multiplication oracle (one-hot / normal encoding, digits 0-9, products 0-81).

One-hot encoding:
  Variables 1-10:   digit 1  (d1_0...d1_9  — exactly one true, value = index)
  Variables 11-20:  digit 2  (d2_0...d2_9  — exactly one true, value = index)
  Variables 21-102: product  (p_0...p_81   — exactly one true, value = index)

The constraint is satisfied iff all three groups are valid one-hot vectors
and the true indices satisfy: i * j == k.

Single-label interface: check(full_assignment) -> bool over all 102 variables.
"""

N_VARS = 102

VARIABLE_NAMES = (
    [f"d1_{i}" for i in range(10)]   # vars 1-10
    + [f"d2_{i}" for i in range(10)] # vars 11-20
    + [f"p_{i}" for i in range(82)]  # vars 21-102
)

VARIABLE_DESCRIPTIONS = (
    [f"the first digit is {i}" for i in range(10)]
    + [f"the second digit is {i}" for i in range(10)]
    + [f"the product is {i}" for i in range(82)]
)

VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [10, 10, 82],
}


from fonesy.utils.one_hot import _int_to_one_hot, _one_hot_to_int


# Valid assignments (d1 * d2 = p), used by ground-truth-dimacs/check.py.
def POSITIVE_SAMPLER(rng):
    """Return a random valid multiplication assignment, encoded as a 102-bit one-hot triple."""
    d1 = rng.randint(0, 9)
    d2 = rng.randint(0, 9)
    p = d1 * d2
    return _int_to_one_hot(d1, 10) + _int_to_one_hot(d2, 10) + _int_to_one_hot(p, 82)


def check(assignment) -> bool:
    """Return True iff the 102-bit assignment encodes a valid multiplication.

    Args:
        assignment: list of 102 int values (0/1).
    """
    d1 = _one_hot_to_int(assignment, 0, 10)
    d2 = _one_hot_to_int(assignment, 10, 10)
    p  = _one_hot_to_int(assignment, 20, 82)
    if d1 < 0 or d2 < 0 or p < 0:
        return False
    return d1 * d2 == p
