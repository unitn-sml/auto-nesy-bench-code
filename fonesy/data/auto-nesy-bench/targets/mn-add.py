"""
MNIST-Addition oracle (one-hot / normal encoding, digits 0-9, sums 0-18).

One-hot encoding:
  Variables 1-10:  digit 1  (d1_0...d1_9 — exactly one true, value = index)
  Variables 11-20: digit 2  (d2_0...d2_9 — exactly one true, value = index)
  Variables 21-39: sum      (s_0...s_18  — exactly one true, value = index)

The constraint is satisfied iff all three groups are valid one-hot vectors
and the true indices satisfy: i + j == k.

Single-label interface: check(full_assignment) -> bool over all 39 variables.
"""

N_VARS = 39

VARIABLE_NAMES = (
    [f"d1_{i}" for i in range(10)]   # vars 1-10
    + [f"d2_{i}" for i in range(10)] # vars 11-20
    + [f"s_{i}" for i in range(19)]  # vars 21-39
)

VARIABLE_DESCRIPTIONS = (
    [f"the first digit is {i}" for i in range(10)]
    + [f"the second digit is {i}" for i in range(10)]
    + [f"the sum is {i}" for i in range(19)]
)

VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [10, 10, 19],
}


from fonesy.utils.one_hot import _int_to_one_hot, _one_hot_to_int


# Valid assignments (d1 + d2 = s), used by ground-truth-dimacs/check.py.
def POSITIVE_SAMPLER(rng):
    """Return a random valid addition assignment, encoded as a 39-bit one-hot triple."""
    d1 = rng.randint(0, 9)
    d2 = rng.randint(0, 9)
    s = d1 + d2
    return _int_to_one_hot(d1, 10) + _int_to_one_hot(d2, 10) + _int_to_one_hot(s, 19)


def check(assignment) -> bool:
    """Return True iff the 39-bit assignment encodes a valid addition.

    Args:
        assignment: list of 39 int values (0/1).
    """
    d1 = _one_hot_to_int(assignment, 0, 10)
    d2 = _one_hot_to_int(assignment, 10, 10)
    s  = _one_hot_to_int(assignment, 20, 19)
    if d1 < 0 or d2 < 0 or s < 0:
        return False
    return d1 + d2 == s
