"""
Sudoku validity oracle (4x4 grid).

One-hot encoding: for each cell (i, j) and digit v (1-4),
  var(i, j, v) = i*16 + j*4 + v   (1-indexed DIMACS variable)

Total: 4 * 4 * 4 = 64 Boolean variables.

A valid assignment encodes a complete Sudoku solution:
  - Each cell holds exactly one digit (one-hot constraint per cell)
  - Each digit 1-4 appears exactly once in every row
  - Each digit 1-4 appears exactly once in every column
  - Each digit 1-4 appears exactly once in every 2x2 box
"""

N_VARS = 64

VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [4] * 16,   # 16 cells, each with a 4-bit one-hot block
}

# A valid board; its 24 digit relabelings are valid boards used as positives
# by ground-truth-dimacs/check.py.
_CANONICAL_SOLUTION = [
    [((c + 2 * (r % 2) + r // 2) % 4) + 1 for c in range(4)] for r in range(4)
]


def POSITIVE_SAMPLER(rng):
    """Return a random valid Sudoku board, encoded as a 64-element assignment."""
    perm = list(range(1, 5))
    rng.shuffle(perm)
    grid = [[perm[v - 1] for v in row] for row in _CANONICAL_SOLUTION]
    a = [0] * 64
    for i in range(4):
        for j in range(4):
            a[i * 16 + j * 4 + (grid[i][j] - 1)] = 1
    return a

VARIABLE_NAMES = [
    f"cell_{i}_{j}_val_{v}"
    for i in range(4)
    for j in range(4)
    for v in range(1, 5)
]

VARIABLE_DESCRIPTIONS = [
    f"cell at row {i}, column {j} contains digit {v}"
    for i in range(4)
    for j in range(4)
    for v in range(1, 5)
]


def check(assignment) -> bool:
    """
    Validate a Sudoku assignment.

    Args:
        assignment: list of 64 ints (0 or 1), where assignment[k] is the
                    truth value of DIMACS variable k+1 (0-indexed).
                    Variable var(i,j,v) = i*16 + j*4 + v maps to
                    assignment[i*16 + j*4 + v - 1].

    Returns:
        True iff the assignment encodes a valid complete Sudoku solution.
    """
    a = list(assignment)

    # Decode grid: grid[i][j] = digit value (1-4), or 0 if the cell is invalid
    grid = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            base = i * 16 + j * 4   # 0-indexed offset for cell (i,j)
            bits = a[base: base + 4]
            if sum(bits) != 1:
                return False
            grid[i][j] = bits.index(1) + 1   # digit 1-4

    digits = set(range(1, 5))

    # Row uniqueness
    for i in range(4):
        if set(grid[i]) != digits:
            return False

    # Column uniqueness
    for j in range(4):
        if {grid[i][j] for i in range(4)} != digits:
            return False

    # 2x2 box uniqueness
    for br in range(2):
        for bc in range(2):
            box = {
                grid[br * 2 + di][bc * 2 + dj]
                for di in range(2)
                for dj in range(2)
            }
            if box != digits:
                return False

    return True
