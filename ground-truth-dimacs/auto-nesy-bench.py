"""Compile the ground-truth CNF of every auto-nesy-bench task with PySAT.

Each ``<task>_cnf()`` encodes the task's constraint directly (one-hot /
exactly-one constraints, arithmetic implications, Tseitin encodings of
Boolean circuits, or blocking clauses over the invalid assignments of a small
domain). ``road-r`` is not compiled: its CNF is the set of requirements
released with ROAD-R. ``check.py`` verifies every target against its oracle.

Usage::

    python ground-truth-dimacs/auto-nesy-bench.py [output_dir]

writes ``<output_dir>/<task>.dimacs`` (``fonesy/data/auto-nesy-bench/targets``
contains the committed copies).
"""

import itertools
import random
from dataclasses import dataclass
from pathlib import Path

from pysat.formula import CNF
from pysat.solvers import Solver as _Solver



# ---------------------------------------------------------------------------
# Shared PySAT helpers
# ---------------------------------------------------------------------------


def _one_hot_constraints(cnf: CNF, groups: list):
    """Add exactly-one (one-hot) constraints for each variable group."""
    for g in groups:
        cnf.append(list(g))                          # at-least-one
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                cnf.append([-g[i], -g[j]])           # at-most-one pair


def _one_hot_iter(block_sizes: list):
    """Yield every valid one-hot assignment for a list of block sizes."""
    options = [
        [tuple(1 if j == i else 0 for j in range(k)) for i in range(k)]
        for k in block_sizes
    ]
    for combo in itertools.product(*options):
        yield [bit for block in combo for bit in block]


def _blocking_clause(assignment: list) -> list:
    """Return the clause falsified by exactly this assignment."""
    return [-(i + 1) if v else (i + 1) for i, v in enumerate(assignment)]


def _concept_guard(bits: list, var_offset: int = 0) -> list:
    """
    Implication guard literals: true when the concept variable block ≠ bits.

    bit=1  → negative literal  (clause is satisfied when that var is 0)
    bit=0  → positive literal  (clause is satisfied when that var is 1)

    When the concept variable block equals bits exactly, all guard literals
    are false and only the clause head decides satisfaction.
    """
    return [
        -(var_offset + i + 1) if b else (var_offset + i + 1)
        for i, b in enumerate(bits)
    ]


def _encode_and_or_output(cnf: CNF, out_var: int, pos_vars: list, neg_vars: list):
    """
    Encode  out_var  ↔  (OR pos_vars) ∧ (¬ each neg_var).

    Forward  (out_var → components):
      [-out_var] + pos_vars          out_var implies at least one pos is true
      [-out_var, -nv]  per nv        out_var implies each neg is false

    Backward (per pos_i: pos_i true and no neg → out_var):
      [out_var, -pos_i] + neg_vars   one clause per pos_var
    """
    cnf.append([-out_var] + list(pos_vars))
    for nv in neg_vars:
        cnf.append([-out_var, -nv])
    for pv in pos_vars:
        cnf.append([out_var, -pv] + list(neg_vars))


def _encode_or_output(cnf: CNF, out_var: int, pos_vars: list):
    """
    Encode  out_var  ↔  OR pos_vars.

    Forward  : [-out_var] + pos_vars
    Backward : [out_var, -pv]  per pv
    """
    cnf.append([-out_var] + list(pos_vars))
    for pv in pos_vars:
        cnf.append([out_var, -pv])


# ---------------------------------------------------------------------------
# Kand-Logic patterns
# ---------------------------------------------------------------------------


def _kand_parse(a: list, offset: int, n_primitives: int = 3):
    """Return (shape_indices, color_indices) for the image starting at offset."""
    shapes, colors = [], []
    for p in range(n_primitives):
        s = a[offset + p * 6:     offset + p * 6 + 3]
        c = a[offset + p * 6 + 3: offset + p * 6 + 6]
        shapes.append(s.index(1) if sum(s) == 1 else -1)
        colors.append(c.index(1) if sum(c) == 1 else -1)
    return shapes, colors


def _diff(lst): return len(set(lst)) == len(lst) and -1 not in lst
def _same(lst): return len(set(lst)) == 1 and -1 not in lst
def _pair(lst): return not _same(lst) and not _diff(lst) and -1 not in lst


def _kand_satisfied(a: list) -> bool:
    s1, c1 = _kand_parse(a, 0)
    s2, c2 = _kand_parse(a, 18)
    if -1 in s1 or -1 in c1 or -1 in s2 or -1 in c2:
        return False
    return (
        (_same(c1) and _same(c2)) or (_pair(c1) and _pair(c2)) or (_diff(c1) and _diff(c2)) or
        (_same(s1) and _same(s2)) or (_pair(s1) and _pair(s2)) or (_diff(s1) and _diff(s2))
    )


def _kand_satisfied_simple(a: list) -> bool:
    """Kand-Logic-Simple: 2 primitives per image, so same/diff are complementary
    (no "pair" case) and the pattern disjunction drops from 6 branches to 4."""
    s1, c1 = _kand_parse(a, 0, n_primitives=2)
    s2, c2 = _kand_parse(a, 12, n_primitives=2)
    if -1 in s1 or -1 in c1 or -1 in s2 or -1 in c2:
        return False
    return (
        (_same(c1) and _same(c2)) or (_diff(c1) and _diff(c2)) or
        (_same(s1) and _same(s2)) or (_diff(s1) and _diff(s2))
    )


# ---------------------------------------------------------------------------
# CEBaB and CIFAR-10 rules
# ---------------------------------------------------------------------------


def _cebab_label(idxs: list) -> int:
    """CEBaB label of 5 concept values (0=Positive, 1=Negative, 2=Neutral,
    3=unknown): 0=positive, 1=negative, 2=neutral, 3=unknown, 4=conflict.
    """
    n_pos = idxs.count(0)
    n_neg = idxs.count(1)
    n_neu = idxs.count(2)
    n_unk = idxs.count(3)
    if n_pos > n_neg and n_pos >= n_neu and n_pos >= n_unk:
        return 0
    if n_neg > n_pos and n_neg >= n_neu and n_neg >= n_unk:
        return 1
    if n_neu > n_pos and n_neu > n_neg and n_neu >= n_unk:
        return 2
    if n_unk > n_pos and n_unk > n_neg and n_unk > n_neu:
        return 3
    return 4  # conflict: N_Pos == N_Neg tied for the overall max


# CIFAR-10 concept-bottleneck: 7-bit concept pattern -> label index, with
# automobile/truck and dog/horse merged and the other bits fixed.
# Order: whl, met, wng, ani, hai, hrn, snt.
_CIFAR10_CANONICAL = {
    (0, 1, 1, 0, 0, 0, 0): 0,  # airplane
    (1, 1, 0, 0, 0, 0, 0): 1,  # car_or_truck
    (0, 0, 1, 1, 0, 0, 0): 2,  # bird
    (0, 0, 0, 1, 0, 0, 0): 3,  # frog
    (0, 0, 0, 1, 1, 1, 1): 4,  # deer
    (0, 0, 0, 1, 1, 0, 0): 5,  # cat
    (0, 0, 0, 1, 1, 0, 1): 6,  # dog_or_horse
    (0, 1, 0, 0, 0, 0, 0): 7,  # ship
}


# ---------------------------------------------------------------------------
# CNF compilers
# ---------------------------------------------------------------------------


def cle4evr_cnf() -> str:
    """
    CLE4EVR (12 vars): two objects share the same shape AND color.

    Vars 1-3  : obj1 shape  (cube / sphere / cylinder)
    Vars 4-6  : obj1 color  (red / blue / green)
    Vars 7-9  : obj2 shape
    Vars 10-12: obj2 color

    Encoding: one-hot for every attribute group plus biconditional equality
    between corresponding attributes of the two objects.
    """
    cnf = CNF()

    _one_hot_constraints(cnf, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # shape1 == shape2  and  color1 == color2
    for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
        cnf.append([-a,  b])   # a → b
        cnf.append([ a, -b])   # b → a

    return cnf.to_dimacs()


def mn_add_bin_cnf() -> str:
    """
    MNIST-Addition (13 vars): two 4-bit digits whose sum is correctly encoded.

    Vars 1-4 : digit 1 bits (LSB first)
    Vars 5-8 : digit 2 bits (LSB first)
    Vars 9-13: sum bits     (LSB first)

    Encoding: for every (d1, d2) ∈ {0…9}², implication clauses force the five
    sum-bit variables to equal the binary encoding of d1 + d2.  Twelve additional
    clauses block encodings 10…15 in each four-bit digit block.
    """
    cnf = CNF()

    for d1 in range(10):
        d1_bits = [(d1 >> i) & 1 for i in range(4)]
        guard_d1 = _concept_guard(d1_bits, var_offset=0)
        for d2 in range(10):
            d2_bits = [(d2 >> i) & 1 for i in range(4)]
            guard = guard_d1 + _concept_guard(d2_bits, var_offset=4)
            s = d1 + d2
            for j in range(5):
                bit = (s >> j) & 1
                out_lit = 9 + j if bit else -(9 + j)
                cnf.append(guard + [out_lit])

    for invalid_digit in range(10, 16):
        bits = [(invalid_digit >> i) & 1 for i in range(4)]
        cnf.append(_concept_guard(bits, var_offset=0))
        cnf.append(_concept_guard(bits, var_offset=4))

    return cnf.to_dimacs()


def mn_mul_bin_cnf() -> str:
    """
    MNIST-Multiplication (15 vars): two 4-bit digits whose product is correctly encoded.

    Vars 1-4 : digit 1 bits (LSB first)
    Vars 5-8 : digit 2 bits (LSB first)
    Vars 9-15: product bits  (LSB first)

    Encoding: same range-restricted implication-clause approach as
    MNIST-Addition, over 7 product bits.
    """
    cnf = CNF()

    for d1 in range(10):
        d1_bits = [(d1 >> i) & 1 for i in range(4)]
        guard_d1 = _concept_guard(d1_bits, var_offset=0)
        for d2 in range(10):
            d2_bits = [(d2 >> i) & 1 for i in range(4)]
            guard = guard_d1 + _concept_guard(d2_bits, var_offset=4)
            p = d1 * d2
            for j in range(7):
                bit = (p >> j) & 1
                out_lit = 9 + j if bit else -(9 + j)
                cnf.append(guard + [out_lit])

    for invalid_digit in range(10, 16):
        bits = [(invalid_digit >> i) & 1 for i in range(4)]
        cnf.append(_concept_guard(bits, var_offset=0))
        cnf.append(_concept_guard(bits, var_offset=4))

    return cnf.to_dimacs()


def mn_add_cnf() -> str:
    """
    MNIST-Addition normal/one-hot encoding (39 vars).

    Vars 1-10 : digit 1 one-hot (d1_0...d1_9)
    Vars 11-20: digit 2 one-hot (d2_0...d2_9)
    Vars 21-39: sum one-hot     (s_0...s_18)

    Encoding:
      - Exactly-one constraints for each group (ALO + pairwise AMO).
      - Implication: d1_i AND d2_j → s_{i+j}  for all i,j in 0..9.
        In clause form: -(i+1) OR -(11+j) OR (21+i+j).

    Total: 46 + 46 + 172 + 100 = 364 clauses.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [list(range(1, 11)), list(range(11, 21)), list(range(21, 40))])
    for i in range(10):
        for j in range(10):
            cnf.append([-(i + 1), -(11 + j), 21 + (i + j)])
    return cnf.to_dimacs()


def mn_mul_cnf() -> str:
    """
    MNIST-Multiplication normal/one-hot encoding (102 vars).

    Vars 1-10  : digit 1 one-hot (d1_0...d1_9)
    Vars 11-20 : digit 2 one-hot (d2_0...d2_9)
    Vars 21-102: product one-hot (p_0...p_81)

    Encoding:
      - Exactly-one constraints for each group (ALO + pairwise AMO).
      - Implication: d1_i AND d2_j → p_{i*j}  for all i,j in 0..9.
        In clause form: -(i+1) OR -(11+j) OR (21+i*j).

    Total: 46 + 46 + 3322 + 100 = 3514 clauses.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [list(range(1, 11)), list(range(11, 21)), list(range(21, 103))])
    for i in range(10):
        for j in range(10):
            cnf.append([-(i + 1), -(11 + j), 21 + (i * j)])
    return cnf.to_dimacs()


def kand_logic_cnf() -> str:
    """
    Kand-Logic (36 vars): two images satisfy a shared color/shape pattern.

    Layout — 6 vars per primitive (3 shape + 3 color), 18 per image:
      Image 1, Prim k (k=1,2,3): vars 6(k-1)+1..3 shape, 6(k-1)+4..6 color
      Image 2, Prim k           : same layout with +18 offset

    Encoding:
      - One-hot constraints for all 12 attribute groups (4 clauses × 12 = 48)
      - Blocking clause for every valid one-hot assignment that does NOT satisfy
        the shared pattern (same / pair / diff on color or shape across both images)
    """
    cnf = CNF()

    groups = [list(range(3 * i + 1, 3 * i + 4)) for i in range(12)]
    _one_hot_constraints(cnf, groups)

    for a in _one_hot_iter([3] * 12):
        if not _kand_satisfied(a):
            cnf.append(_blocking_clause(a))

    return cnf.to_dimacs()


def kand_logic_2_cnf() -> str:
    """
    Kand-Logic-Simple (24 vars): reduced Kand-Logic with 2 primitives per
    image instead of 3 (no "pair" predicate — same/diff are complementary).

    Layout — 6 vars per primitive, 12 per image:
      Image 1, Prim k (k=1,2): vars 6(k-1)+1..3 shape, 6(k-1)+4..6 color
      Image 2, Prim k        : same layout with +12 offset

    Encoding:
      - One-hot constraints for all 8 attribute groups (4 clauses × 8 = 32)
      - Blocking clause for every valid one-hot assignment that does NOT
        satisfy the shared pattern (same / diff on color or shape across
        both images)
    """
    cnf = CNF()

    groups = [list(range(3 * i + 1, 3 * i + 4)) for i in range(8)]
    _one_hot_constraints(cnf, groups)

    for a in _one_hot_iter([3] * 8):
        if not _kand_satisfied_simple(a):
            cnf.append(_blocking_clause(a))

    return cnf.to_dimacs()


def bdd_oia_cnf() -> str:
    """
    BDD-OIA (25 vars): road-driving action constraints.

    Concepts 1-21:
      green_light(1)  follow(2)        road_clear(3)
      red_light(4)    traffic_sign(5)  car(6)   person(7)  rider(8)  other_obstacle(9)
      left_lane(10)   left_green_light(11)  left_follow(12)
      no_left_lane(13) left_obstacle(14)   left_solid_line(15)
      right_lane(16)  right_green_light(17) right_follow(18)
      no_right_lane(19) right_obstacle(20)  right_solid_line(21)
    Actions 22-25:
      move_forward(22)  stop_action(23)  turn_left(24)  turn_right(25)

    Encoding: direct Tseitin circuit over 4 output bits (~31 clauses, no auxiliary vars).

      move_forward ↔ (v1∨v2∨v3) ∧ ¬(v4∨v5∨v6∨v7∨v8∨v9)
      stop_action  ↔  v4∨v5∨v6∨v7∨v8∨v9
      turn_left    ↔ (v10∨v11∨v12) ∧ ¬(v13∨v14∨v15)
      turn_right   ↔ (v16∨v17∨v18) ∧ ¬(v19∨v20∨v21)
    """
    cnf = CNF()

    _encode_and_or_output(cnf, 22, [1, 2, 3],        [4, 5, 6, 7, 8, 9])
    _encode_or_output(    cnf, 23, [4, 5, 6, 7, 8, 9])
    _encode_and_or_output(cnf, 24, [10, 11, 12],     [13, 14, 15])
    _encode_and_or_output(cnf, 25, [16, 17, 18],     [19, 20, 21])

    return cnf.to_dimacs()


def bdd_oia_2_cnf() -> str:
    """
    BDD-OIA-Simple (11 vars): road-driving forward/stop constraints only.

    A reduced BDD-OIA: the turn-left/turn-right concepts and actions are
    dropped entirely, keeping only the concepts and rules that determine
    move_forward/stop_action.

    Concepts 1-9:
      green_light(1)  follow(2)        road_clear(3)
      red_light(4)    traffic_sign(5)  car(6)   person(7)  rider(8)  other_obstacle(9)
    Actions 10-11:
      move_forward(10)  stop_action(11)

    Encoding: direct Tseitin circuit over 2 output bits — the same
    move_forward/stop_action sub-circuit used by bdd_oia_cnf(), renumbered
    onto 11 variables instead of 25.

      move_forward ↔ (v1∨v2∨v3) ∧ ¬(v4∨v5∨v6∨v7∨v8∨v9)
      stop_action  ↔  v4∨v5∨v6∨v7∨v8∨v9
    """
    cnf = CNF()

    _encode_and_or_output(cnf, 10, [1, 2, 3],        [4, 5, 6, 7, 8, 9])
    _encode_or_output(    cnf, 11, [4, 5, 6, 7, 8, 9])

    return cnf.to_dimacs()


def sudoku_cnf() -> str:
    """
    Sudoku validity (64 vars): a complete 4x4 Sudoku solution.

    Variable layout — one-hot encoding over 16 cells × 4 values:
      var(i, j, v) = i*16 + j*4 + v  (1-indexed, i,j ∈ [0,3], v ∈ [1,4])

    Constraints:
      - Each cell holds exactly one digit (at-least-one + pairwise at-most-one)
      - Each digit appears at most once per row   (pairwise)
      - Each digit appears at most once per column (pairwise)
      - Each digit appears at most once per 2x2 box (pairwise)
    """
    cnf = CNF()

    def var(i, j, v):
        return i * 16 + j * 4 + v

    for i in range(4):
        for j in range(4):
            cnf.append([var(i, j, v) for v in range(1, 5)])
            for v in range(1, 5):
                for w in range(v + 1, 5):
                    cnf.append([-var(i, j, v), -var(i, j, w)])

    for v in range(1, 5):
        for i in range(4):
            for j in range(4):
                for k in range(j + 1, 4):
                    cnf.append([-var(i, j, v), -var(i, k, v)])
        for j in range(4):
            for i in range(4):
                for k in range(i + 1, 4):
                    cnf.append([-var(i, j, v), -var(k, j, v)])
        for br in range(2):
            for bc in range(2):
                cells = [(br * 2 + di, bc * 2 + dj) for di in range(2) for dj in range(2)]
                for ci in range(len(cells)):
                    for ck in range(ci + 1, len(cells)):
                        cnf.append([-var(*cells[ci], v), -var(*cells[ck], v)])

    return cnf.to_dimacs()


def sushi_cnf() -> str:
    """
    Sushi permutation validity (16 vars): a valid 4x4 assignment/ranking
    permutation matrix — the "sushi" example in Xu et al., "A Semantic Loss
    Function for Deep Learning with Symbolic Knowledge", ICML 2018.

    Variable layout — one-hot encoding over 4 items x 4 ranks:
      var(i, j) = i*4 + j + 1  (1-indexed, i,j in [0,3])

    Constraints:
      - Each item is assigned exactly one rank   (one-hot per row)
      - Each rank is occupied by exactly one item (one-hot per column)
    """
    cnf = CNF()

    def var(i, j):
        return i * 4 + j + 1

    _one_hot_constraints(cnf, [[var(i, j) for j in range(4)] for i in range(4)])
    _one_hot_constraints(cnf, [[var(i, j) for i in range(4)] for j in range(4)])

    return cnf.to_dimacs()


def fashion_cnf() -> str:
    """
    Fashion-Mut-Ex (10 vars): exactly-one mutual-exclusion constraint over the
    10 Fashion-MNIST/MNIST/CIFAR-10-style class indicators (Section 4's
    semi-supervised classification constraint, generalized from the paper's
    n=3 illustrative example to n=10).

    Variable layout: variable i+1 is the one-hot indicator for class i.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [list(range(1, 11))])
    return cnf.to_dimacs()


def chx_cnf() -> str:
    """
    CHX (9 vars): 5-way severity code derived from 4 binary findings.

    Vars 1-4: fracture, pneumothorax, airspace_opacity, nodule_mass (concepts)
    Vars 5-9: healthy, green_code, yellow_code, orange_code, red_code (one-hot label)

    Encoding: one-hot constraint on the label block, plus one implication
    clause per concept combination (2^4 = 16) forcing the label bit at
    index popcount(concepts) — mirrors mn_add_bin_cnf()'s guard-
    enumeration style.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [[5, 6, 7, 8, 9]])
    for bits in itertools.product((0, 1), repeat=4):
        guard = _concept_guard(list(bits))
        label_var = 5 + sum(bits)
        cnf.append(guard + [label_var])
    return cnf.to_dimacs()


def cebab_cnf() -> str:
    """
    CEBaB (25 vars): overall sentiment label derived from 5 categorical
    aspect-majority concepts (4-valued one-hot each).

    Vars 1-20: 5 concept blocks of 4 (Positive/Negative/Neutral/unknown)
    Vars 21-25: label, one-hot (positive/negative/neutral/unknown/conflict)

    Encoding: one-hot constraints on all 5 concept blocks + the label
    block, plus one implication clause per valid concept combination
    (4^5 = 1024) forcing the correct label bit — guard is the negation of
    the 5 active one-hot literals (mn_add_cnf()'s idiom),
    label derived via the independently-transcribed _cebab_label().
    """
    cnf = CNF()
    groups = [list(range(4 * k + 1, 4 * k + 5)) for k in range(5)]
    _one_hot_constraints(cnf, groups)
    _one_hot_constraints(cnf, [[21, 22, 23, 24, 25]])

    for a in _one_hot_iter([4, 4, 4, 4, 4]):
        active_lits = [4 * k + a[4 * k:4 * k + 4].index(1) + 1 for k in range(5)]
        idxs = [lit - 1 - 4 * k for k, lit in enumerate(active_lits)]
        label = _cebab_label(idxs)
        guard = [-lit for lit in active_lits]
        cnf.append(guard + [21 + label])

    return cnf.to_dimacs()


def cifar10_cnf() -> str:
    """
    CIFAR-10 concept-bottleneck (15 vars): 8-way image class derived from 7
    binary visual concepts, with automobile/truck and dog/horse merged (see
    targets/cifar10.py's docstring for the full canonical-pattern rationale).

    Vars 1-7: whl, met, wng, ani, hai, hrn, snt (concepts, free/independent)
    Vars 8-15: label, one-hot (airplane, car_or_truck, bird, frog, deer,
               cat, dog_or_horse, ship)

    Encoding: one-hot constraint on the label block only. For each of the
    8 canonical concept patterns, a guard clause forces the matching label
    bit (_concept_guard(), as in chx_cnf()). For the other 120 (of 2^7)
    concept patterns, a single blocking clause over the concept vars alone
    forbids that exact pattern regardless of any label choice — the same
    _blocking_clause() helper kand_logic_cnf() uses for invalid one-hot
    combos, applied here to concept-only sub-patterns.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [list(range(8, 16))])

    for bits in itertools.product((0, 1), repeat=7):
        pattern = list(bits)
        label_idx = _CIFAR10_CANONICAL.get(tuple(bits))
        if label_idx is not None:
            guard = _concept_guard(pattern)
            cnf.append(guard + [8 + label_idx])
        else:
            cnf.append(_blocking_clause(pattern))

    return cnf.to_dimacs()


def cifar100_cnf() -> str:
    """
    CIFAR100-Mut-Ex (100 vars): exactly-one mutual-exclusion constraint over
    100 class indicators — same constraint as fashion_cnf(), just at
    n=100 instead of n=10 (Section 4's semi-supervised classification
    constraint, generalized from the paper's n=3 illustrative example).

    Variable layout: variable i+1 is the one-hot indicator for class i.
    """
    cnf = CNF()
    _one_hot_constraints(cnf, [list(range(1, 101))])
    return cnf.to_dimacs()


# ---------------------------------------------------------------------------
# Warcraft grid graph
# ---------------------------------------------------------------------------


def _warcraft_path_hvar(r: int, c: int) -> int:
    return r * 3 + c + 1


def _warcraft_path_vvar(r: int, c: int) -> int:
    return 12 + r * 4 + c + 1


def _warcraft_path_adjacency(n: int = 4) -> dict:
    """Return {(r,c): [(neighbor, edge_var), ...]} for the n x n grid graph."""
    adj = {(r, c): [] for r in range(n) for c in range(n)}
    for r in range(n):
        for c in range(n - 1):
            var = _warcraft_path_hvar(r, c)
            adj[(r, c)].append(((r, c + 1), var))
            adj[(r, c + 1)].append(((r, c), var))
    for r in range(n - 1):
        for c in range(n):
            var = _warcraft_path_vvar(r, c)
            adj[(r, c)].append(((r + 1, c), var))
            adj[(r + 1, c)].append(((r, c), var))
    return adj


def _warcraft_path_cycles(n: int = 4) -> set:
    """
    Enumerate every simple cycle in the n x n grid graph, as a frozenset of
    edge variables.

    Plain exhaustive DFS from every vertex — networkx isn't a project
    dependency, so this doesn't use it. Each geometric cycle is discovered
    once per starting vertex and per traversal direction; deduplication is
    by edge-variable frozenset, so the search itself doesn't need to be
    duplicate-free. Only tractable because the grid is tiny (16 vertices for
    n=4): do not reuse this for larger n. Cycle count explodes combinatorially
    with grid size (4x4: 213, 5x5: 9349, 6x6: 1222363 — see
    warcraft_cnf()'s docstring), which is exactly why this oracle is
    sized at 4x4 rather than the paper's 12x12/6x6.
    """
    adj = _warcraft_path_adjacency(n)
    found = set()

    def dfs(start, current, visited, path_vars):
        for nbr, var in adj[current]:
            if nbr == start and len(visited) >= 3:
                found.add(frozenset(path_vars + [var]))
            elif nbr not in visited:
                visited.add(nbr)
                dfs(start, nbr, visited, path_vars + [var])
                visited.remove(nbr)

    for start in adj:
        dfs(start, start, {start}, [])
    return found


def warcraft_cnf() -> str:
    """Warcraft (24 vars): one simple path from cell (0, 0) to cell (3, 3) of a 4x4 grid.

    Vars 1-12 : edge_right(r,c), r=0..3, c=0..2 — edge (r,c)-(r,c+1)
    Vars 13-24: edge_down(r,c),  r=0..2, c=0..3 — edge (r,c)-(r+1,c)

    Encoding: exactly one selected edge at the two corners, 0 or 2 at every
    other cell (blocking clauses), and one blocking clause per simple cycle of
    the grid. The result has 184 models, the simple corner-to-corner paths.
    """
    cnf = CNF()
    adj = _warcraft_path_adjacency(4)
    incident = {v: [var for _, var in adj[v]] for v in adj}

    s, t = (0, 0), (3, 3)
    for v, inc in incident.items():
        if v in (s, t):
            cnf.append(list(inc))
            for a, b in itertools.combinations(inc, 2):
                cnf.append([-a, -b])
        else:
            k = len(inc)
            for bits in itertools.product((0, 1), repeat=k):
                if sum(bits) not in (0, 2):
                    cnf.append([-inc[i] if bits[i] else inc[i] for i in range(k)])

    for cycle in _warcraft_path_cycles(4):
        cnf.append([-v for v in cycle])

    return cnf.to_dimacs()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_REGISTRY: dict = {
    "bdd-oia": bdd_oia_cnf,
    "bdd-oia-2": bdd_oia_2_cnf,
    "cebab": cebab_cnf,
    "chx": chx_cnf,
    "cifar10": cifar10_cnf,
    "cifar100": cifar100_cnf,
    "cle4evr": cle4evr_cnf,
    "fashion": fashion_cnf,
    "kand-logic": kand_logic_cnf,
    "kand-logic-2": kand_logic_2_cnf,
    "mn-add": mn_add_cnf,
    "mn-add-bin": mn_add_bin_cnf,
    "mn-mul": mn_mul_cnf,
    "mn-mul-bin": mn_mul_bin_cnf,
    "sudoku": sudoku_cnf,
    "sushi": sushi_cnf,
    "warcraft": warcraft_cnf,
}
PROGRAM_NAMES = sorted(_REGISTRY)


def compile_program_to_cnf(name: str) -> str:
    """DIMACS of the ground-truth CNF of task *name* (one of ``PROGRAM_NAMES``)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown program {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def compile_all_programs(output_dir: str = None) -> dict:
    """Compile every task, writing ``<output_dir>/<task>.dimacs`` if given."""
    results = {}
    for name in PROGRAM_NAMES:
        results[name] = _REGISTRY[name]()
        if output_dir is not None:
            path = Path(output_dir) / f"{name}.dimacs"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(results[name])
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile the auto-nesy-bench ground-truth CNFs."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory to write <name>.dimacs files (optional).",
    )
    args = parser.parse_args()

    print(f"{'task':<25}  {'header':<20}  clauses")
    print("-" * 55)
    for prog_name, dimacs in compile_all_programs(args.output_dir).items():
        lines = dimacs.strip().splitlines()
        header = next(l for l in lines if l.startswith("p cnf"))
        n_clauses = sum(1 for l in lines if l and not l.startswith(("p", "c")))
        print(f"{prog_name:<25}  {header:<20}  {n_clauses}")
    if args.output_dir:
        print(f"\nFiles written to: {args.output_dir}")
