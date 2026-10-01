"""
Warcraft-Path oracle (4x4 grid).

A single simple path across a 4x4 grid of cells, from the top-left cell
(0,0) to the bottom-right cell (3,3) -- the grid-connectivity constraint
behind the Semantic Probabilistic Layers paper's "Simple Path" and "Warcraft
Shortest Path" benchmarks (Sec. 5), at a scale (4x4) chosen so the ground-
truth CNF (see ground-truth-dimacs/auto-nesy-bench.py) can encode "no
disconnected loop elsewhere in the grid" via a flat, tractable set of
blocking clauses. Structural validity only -- no terrain weights/cost, like
every other program oracle in this repo.

Variable layout -- one boolean per grid edge:
  edge_right(r, c) = r*3 + c + 1        (r=0..3, c=0..2, horizontal edges)
  edge_down(r, c)  = 12 + r*4 + c + 1   (r=0..2, c=0..3, vertical edges)

Total: 24 Boolean variables.

A valid assignment selects a set of edges forming a single simple path from
(0,0) to (3,3):
  - (0,0) and (3,3) each have exactly one selected incident edge.
  - every other cell has 0 or 2 selected incident edges.
  - the selected edges contain no extra, disconnected cycle.
There are exactly 184 valid assignments (the number of simple paths between
opposite corners of a 4x4 grid graph).
"""

N_VARS = 24

_N = 4
_SOURCE = (0, 0)
_SINK = (_N - 1, _N - 1)


def _hvar(r, c):
    return r * 3 + c + 1


def _vvar(r, c):
    return 12 + r * 4 + c + 1


def _build_adjacency():
    adj = {(r, c): [] for r in range(_N) for c in range(_N)}
    for r in range(_N):
        for c in range(_N - 1):
            var = _hvar(r, c)
            adj[(r, c)].append(((r, c + 1), var))
            adj[(r, c + 1)].append(((r, c), var))
    for r in range(_N - 1):
        for c in range(_N):
            var = _vvar(r, c)
            adj[(r, c)].append(((r + 1, c), var))
            adj[(r + 1, c)].append(((r, c), var))
    return adj


_ADJ = _build_adjacency()

VARIABLE_NAMES = [f"edge_right_{r}_{c}" for r in range(_N) for c in range(_N - 1)] + [
    f"edge_down_{r}_{c}" for r in range(_N - 1) for c in range(_N)
]

VARIABLE_DESCRIPTIONS = [
    f"the edge between cells ({r},{c}) and ({r},{c + 1}) is part of the path"
    for r in range(_N)
    for c in range(_N - 1)
] + [
    f"the edge between cells ({r},{c}) and ({r + 1},{c}) is part of the path"
    for r in range(_N - 1)
    for c in range(_N)
]


def _enumerate_all_paths():
    """Enumerate every simple path from _SOURCE to _SINK as a list of used-edge-var lists."""
    paths = []

    def dfs(current, visited, path_vars):
        if current == _SINK:
            paths.append(list(path_vars))
            return
        for nbr, var in _ADJ[current]:
            if nbr not in visited:
                visited.add(nbr)
                dfs(nbr, visited, path_vars + [var])
                visited.remove(nbr)

    dfs(_SOURCE, {_SOURCE}, [])
    return paths


# Precomputed once at import time (184 paths on a 4x4 grid -- cheap).
_ALL_PATHS = _enumerate_all_paths()


def POSITIVE_SAMPLER(rng):
    """Return a random valid warcraft assignment (a uniformly random simple s-t path).

    Needed because independent-uniform bit sampling essentially never lands
    on a valid path (184 valid assignments out of 2**24), mirroring
    sudoku.py's/road_r.py's POSITIVE_SAMPLER.
    """
    path_vars = rng.choice(_ALL_PATHS)
    a = [0] * N_VARS
    for var in path_vars:
        a[var - 1] = 1
    return a


def check(assignment) -> bool:
    """
    Validate a warcraft assignment.

    Args:
        assignment: list of 24 ints (0 or 1), where assignment[k] is the
                    truth value of DIMACS variable k+1 (0-indexed).

    Returns:
        True iff the selected edges form a single simple path from (0,0) to
        (3,3), with no extra disconnected component.
    """
    a = list(assignment)

    deg = {v: 0 for v in _ADJ}
    used_from = {v: [] for v in _ADJ}
    for v in _ADJ:
        for nbr, var in _ADJ[v]:
            if a[var - 1] == 1:
                deg[v] += 1
                used_from[v].append(nbr)

    if deg[_SOURCE] != 1 or deg[_SINK] != 1:
        return False
    for v, d in deg.items():
        if v in (_SOURCE, _SINK):
            continue
        if d not in (0, 2):
            return False

    total_used = sum(a)

    # Deterministic walk from the source: at each step follow the one used
    # edge that doesn't lead back where we came from.
    prev = None
    cur = _SOURCE
    steps = 0
    while cur != _SINK:
        candidates = [nbr for nbr in used_from[cur] if nbr != prev]
        if len(candidates) != 1:
            return False
        prev, cur = cur, candidates[0]
        steps += 1
        if steps > total_used:
            return False

    # steps == total_used rules out an extra, disconnected cycle elsewhere --
    # degree constraints alone can't catch that, since every vertex on a
    # stray cycle still has degree 2.
    return steps == total_used
