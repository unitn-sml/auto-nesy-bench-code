"""
Fashion-Mut-Ex oracle (10 vars): exactly-one mutual-exclusion constraint over
the 10 Fashion-MNIST class indicators - the semi-supervised classification
constraint from Section 4 (MNIST / Fashion-MNIST / CIFAR-10 all share it),
generalized from the paper's illustrative n=3 example
((x1 v x2 v x3) & (not x1 v not x2) & (not x1 v not x3) & (not x2 v not x3))
to n=10.

One-hot encoding: variable i+1 (1-indexed DIMACS) is True iff the image
belongs to Fashion-MNIST class i (Zalando's canonical label order).

Total: 10 Boolean variables.
"""

N_VARS = 10

VARIABLE_NAMES = [
    "class_0_tshirt_top",
    "class_1_trouser",
    "class_2_pullover",
    "class_3_dress",
    "class_4_coat",
    "class_5_sandal",
    "class_6_shirt",
    "class_7_sneaker",
    "class_8_bag",
    "class_9_ankle_boot",
]

VARIABLE_DESCRIPTIONS = [
    "the clothing item is a t-shirt",
    "the clothing item is a pair of trousers",
    "the clothing item is a pullover",
    "the clothing item is a dress",
    "the clothing item is a coat",
    "the footwear is a sandal",
    "the clothing item is a shirt",
    "the footwear is a sneaker",
    "the item is a bag",
    "the footwear is an ankle boot",
]

VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [10],
}


def check(assignment) -> bool:
    """
    assignment[i] = 0 or 1 for DIMACS variable i+1.
    Returns True iff exactly one of the 10 class indicators is True.
    """
    return sum(assignment) == 1
