"""
CIFAR-10 concept-bottleneck oracle: assign one of 8 image-class labels from
7 binary visual concepts (designed concepts, deterministically derived from
the ground-truth class — not learned/annotated independently).

Whole-assignment-validity interface (no N_CONCEPT_VARS): the oracle checks
the full 15-variable assignment (7 concepts + 8-way one-hot label) at once,
since a concept vector can have NO valid label (concept vectors outside the
8 canonical patterns below never occur for any real class).

Variable layout:
  Variables 1-7 (concepts): whl (has wheels), met (is metallic), wng (has
    wings), ani (is an animal), hai (is hairy), hrn (has horns), snt (has a
    snout)
  Variables 8-15 (label, one-hot): airplane, car_or_truck, bird, frog, deer,
    cat, dog_or_horse, ship

The original per-class rules only pin a SUBSET of the 7 concepts each (e.g.
bird only requires ani/¬met/wng), which would let some concept vectors
satisfy more than one unrelated class's rule. To keep the concept->label
mapping a clean one-to-one partition, automobile and truck are merged into
a single label (car_or_truck) and dog and horse into a single label
(dog_or_horse) — matching the original rules' stated automobile-or-truck
and dog-or-horse pairing — and every other previously-free bit is pinned to
a canonical value, giving 8 fully-specified, mutually disjoint 7-bit
patterns (order whl/met/wng/ani/hai/hrn/snt):
  airplane:      met, wng, not ani                         -> 0,1,1,0,0,0,0
  car_or_truck:  whl, met, not wng, not ani                -> 1,1,0,0,0,0,0
  bird:          ani, wng, not met                         -> 0,0,1,1,0,0,0
  frog:          ani, not met, not hai, not wng, not hrn, not snt -> 0,0,0,1,0,0,0
  deer:          ani, hai, hrn, snt, not met, not wng       -> 0,0,0,1,1,1,1
  cat:           ani, hai, not met, not wng, not hrn, not snt -> 0,0,0,1,1,0,0
  dog_or_horse:  ani, hai, snt, not met, not wng, not hrn    -> 0,0,0,1,1,0,1
  ship:          met, not ani, not whl, not wng              -> 0,1,0,0,0,0,0

A concept vector not matching any of the 8 rows above has NO valid label —
every one-hot label choice is invalid for it.
"""

N_VARS = 15

VARIABLE_NAMES = [
    "whl", "met", "wng", "ani", "hai", "hrn", "snt",  # vars 1-7 (concepts)
    "label_airplane", "label_car_or_truck", "label_bird", "label_frog",  # vars 8-11
    "label_deer", "label_cat", "label_dog_or_horse", "label_ship",       # vars 12-15
]

VARIABLE_DESCRIPTIONS = [
    "the object has wheels",
    "the object is metallic",
    "the object has wings",
    "the object is an animal",
    "the object is hairy",
    "the object has horns",
    "the object has a snout",
    "the object is an airplane",
    "the object is either a car or a truck",
    "the object is a bird",
    "the object is a frog",
    "the object is a deer",
    "the object is a cat",
    "the object is either a dog or a horse",
    "the object is a ship",
]

# Concepts are free Booleans; only the label block (vars 8-15) is one-hot.
VALIDITY_SPEC = None

_CANONICAL_PATTERNS = {
    0: (0, 1, 1, 0, 0, 0, 0),  # airplane
    1: (1, 1, 0, 0, 0, 0, 0),  # car_or_truck
    2: (0, 0, 1, 1, 0, 0, 0),  # bird
    3: (0, 0, 0, 1, 0, 0, 0),  # frog
    4: (0, 0, 0, 1, 1, 1, 1),  # deer
    5: (0, 0, 0, 1, 1, 0, 0),  # cat
    6: (0, 0, 0, 1, 1, 0, 1),  # dog_or_horse
    7: (0, 1, 0, 0, 0, 0, 0),  # ship
}


def check(assignment) -> bool:
    """
    assignment[i] = 0 or 1 for DIMACS variable i+1.

    Returns True iff the 7 concept bits (indices 0-6) exactly match one of
    the 8 canonical patterns AND the asserted one-hot label (indices 7-14)
    is that pattern's corresponding class.
    """
    a = list(assignment)
    concept = tuple(int(bool(b)) for b in a[0:7])
    label_block = a[7:15]

    if sum(label_block) != 1:
        return False
    label_idx = label_block.index(1)

    expected_pattern = _CANONICAL_PATTERNS.get(label_idx)
    return expected_pattern is not None and concept == expected_pattern
