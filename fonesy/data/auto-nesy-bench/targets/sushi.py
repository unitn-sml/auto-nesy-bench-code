"""
Sushi permutation oracle (N=4): validity of a 4x4 permutation/assignment
matrix — every item is assigned to exactly one rank and every rank is
occupied by exactly one item.

One-hot encoding: for item i (0-3) and rank j (0-3),
  var(i, j) = i*4 + j + 1   (1-indexed DIMACS variable)

Total: 4 * 4 = 16 Boolean variables.

Reference: the "sushi" ranking example in Xu et al., "A Semantic Loss
Function for Deep Learning with Symbolic Knowledge", ICML 2018.
"""

N_VARS = 16

VARIABLE_NAMES = [
    f"item_{i}_rank_{j}"
    for i in range(4)
    for j in range(4)
]

VARIABLE_DESCRIPTIONS = [
    f"sushi item {i} is assigned to rank {j}"
    for i in range(4)
    for j in range(4)
]

VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [4] * 4,   # 4 items, each with a 4-bit one-hot block over ranks
}


def check(assignment) -> bool:
    """
    Validate a permutation-matrix assignment.

    Args:
        assignment: list of 16 ints (0 or 1); assignment[k] is the truth
                    value of DIMACS variable k+1. var(i,j) = i*4+j+1 maps
                    to assignment[i*4+j].

    Returns:
        True iff the assignment is a valid 4x4 permutation: every item has
        exactly one rank and all 4 ranks used are distinct (equivalently,
        every rank is occupied by exactly one item).
    """
    a = list(assignment)
    ranks = []
    for i in range(4):
        row = a[i * 4:(i + 1) * 4]
        if sum(row) != 1:
            return False
        ranks.append(row.index(1))
    return len(set(ranks)) == 4
