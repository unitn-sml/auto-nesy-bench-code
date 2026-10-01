"""
CLE4EVR oracle — two objects share the same shape AND color.

Variable layout (one-hot encoding):
  Object 1:
    vars 1-3  shape (cube, sphere, cylinder)
    vars 4-6  color (red, blue, green)
  Object 2:
    vars 7-9  shape
    vars 10-12 color

The constraint is satisfied iff:
  - Each one-hot group is valid (exactly one bit is True)
  - Object 1 and Object 2 have the same shape AND the same color

Reference: CLE4EVR dataset (RSBench, Bortolotti et al. 2024)
"""

N_VARS = 12

VARIABLE_NAMES = [
    "obj1_shape_cube", "obj1_shape_sphere", "obj1_shape_cylinder",
    "obj1_color_red",  "obj1_color_blue",   "obj1_color_green",
    "obj2_shape_cube", "obj2_shape_sphere", "obj2_shape_cylinder",
    "obj2_color_red",  "obj2_color_blue",   "obj2_color_green",
]

VARIABLE_DESCRIPTIONS = [
    "the first object is a cube",
    "the first object is a sphere",
    "the first object is a cylinder",
    "the first object is red",
    "the first object is blue",
    "the first object is green",
    "the second object is a cube",
    "the second object is a sphere",
    "the second object is a cylinder",
    "the second object is red",
    "the second object is blue",
    "the second object is green",
]

# 2 objects × 2 attributes (shape + color) = 4 blocks of 3 bits each.
VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [3, 3, 3, 3],
}


def check(assignment) -> bool:
    """
    assignment[i] = 0 or 1 for DIMACS variable i+1.
    Returns True iff both objects share the same shape AND color.
    """
    shape1 = list(assignment[0:3])
    color1 = list(assignment[3:6])
    shape2 = list(assignment[6:9])
    color2 = list(assignment[9:12])

    # Validity: each group must be a proper one-hot
    if sum(shape1) != 1 or sum(color1) != 1 or sum(shape2) != 1 or sum(color2) != 1:
        return False

    return shape1.index(1) == shape2.index(1) and color1.index(1) == color2.index(1)
