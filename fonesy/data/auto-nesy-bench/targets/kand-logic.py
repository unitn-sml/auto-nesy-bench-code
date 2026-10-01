"""
Kand-Logic oracle — full pattern over two images.

Each image contains 3 primitives; each primitive has a shape and a color,
both encoded as 3-way one-hot vectors.

Variable layout (6 vars per primitive, 18 per image, 36 total):

  Image 1
    Primitive 1:  vars  1-3  shape (square, triangle, circle)
                  vars  4-6  color (red, blue, yellow)
    Primitive 2:  vars  7-9  shape
                  vars 10-12 color
    Primitive 3:  vars 13-15 shape
                  vars 16-18 color
  Image 2
    Primitive 1:  vars 19-21 shape
                  vars 22-24 color
    Primitive 2:  vars 25-27 shape
                  vars 28-30 color
    Primitive 3:  vars 31-33 shape
                  vars 34-36 color

Pattern (shared_pattern):
  (same_c(f1) ∧ same_c(f2)) ∨ (pair_c(f1) ∧ pair_c(f2)) ∨ (diff_c(f1) ∧ diff_c(f2))
  ∨
  (same_s(f1) ∧ same_s(f2)) ∨ (pair_s(f1) ∧ pair_s(f2)) ∨ (diff_s(f1) ∧ diff_s(f2))

where:
  diff_c(f) = all 3 primitives have different colors
  same_c(f) = all 3 primitives have the same color
  pair_c(f) = ¬same_c(f) ∧ ¬diff_c(f)   (exactly two share a color)
  diff_s, same_s, pair_s: same for shapes

Reference: Kand-Logic dataset (RSBench, Bortolotti et al. 2024)
"""

N_VARS = 36

VARIABLE_NAMES = [
    # Image 1
    "im1_p1_shape_square", "im1_p1_shape_triangle", "im1_p1_shape_circle",
    "im1_p1_color_red",    "im1_p1_color_blue",     "im1_p1_color_yellow",
    "im1_p2_shape_square", "im1_p2_shape_triangle", "im1_p2_shape_circle",
    "im1_p2_color_red",    "im1_p2_color_blue",     "im1_p2_color_yellow",
    "im1_p3_shape_square", "im1_p3_shape_triangle", "im1_p3_shape_circle",
    "im1_p3_color_red",    "im1_p3_color_blue",     "im1_p3_color_yellow",
    # Image 2
    "im2_p1_shape_square", "im2_p1_shape_triangle", "im2_p1_shape_circle",
    "im2_p1_color_red",    "im2_p1_color_blue",     "im2_p1_color_yellow",
    "im2_p2_shape_square", "im2_p2_shape_triangle", "im2_p2_shape_circle",
    "im2_p2_color_red",    "im2_p2_color_blue",     "im2_p2_color_yellow",
    "im2_p3_shape_square", "im2_p3_shape_triangle", "im2_p3_shape_circle",
    "im2_p3_color_red",    "im2_p3_color_blue",     "im2_p3_color_yellow",
]

VARIABLE_DESCRIPTIONS = [
    "primitive 1 in image 1 is a square",
    "primitive 1 in image 1 is a triangle",
    "primitive 1 in image 1 is a circle",
    "primitive 1 in image 1 is red",
    "primitive 1 in image 1 is blue",
    "primitive 1 in image 1 is yellow",
    "primitive 2 in image 1 is a square",
    "primitive 2 in image 1 is a triangle",
    "primitive 2 in image 1 is a circle",
    "primitive 2 in image 1 is red",
    "primitive 2 in image 1 is blue",
    "primitive 2 in image 1 is yellow",
    "primitive 3 in image 1 is a square",
    "primitive 3 in image 1 is a triangle",
    "primitive 3 in image 1 is a circle",
    "primitive 3 in image 1 is red",
    "primitive 3 in image 1 is blue",
    "primitive 3 in image 1 is yellow",
    "primitive 1 in image 2 is a square",
    "primitive 1 in image 2 is a triangle",
    "primitive 1 in image 2 is a circle",
    "primitive 1 in image 2 is red",
    "primitive 1 in image 2 is blue",
    "primitive 1 in image 2 is yellow",
    "primitive 2 in image 2 is a square",
    "primitive 2 in image 2 is a triangle",
    "primitive 2 in image 2 is a circle",
    "primitive 2 in image 2 is red",
    "primitive 2 in image 2 is blue",
    "primitive 2 in image 2 is yellow",
    "primitive 3 in image 2 is a square",
    "primitive 3 in image 2 is a triangle",
    "primitive 3 in image 2 is a circle",
    "primitive 3 in image 2 is red",
    "primitive 3 in image 2 is blue",
    "primitive 3 in image 2 is yellow",
]

# 2 images × 3 primitives × 2 attributes (shape + color) = 12 blocks of 3 bits each.
# Under uniform sampling, P(valid) = (1/3)^12 ≈ 1/531k — too rare to sample usefully.
VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [3] * 12,
}


def _parse_image(a, offset):
    """
    Extract (shapes, colors) from a 18-bit block starting at `offset`.
    Each is a list of 3 one-hot indices (0/1/2).
    Returns shape_indices and color_indices as lists of int.
    """
    primitives_shape = []
    primitives_color = []
    for p in range(3):
        s_bits = a[offset + p * 6: offset + p * 6 + 3]
        c_bits = a[offset + p * 6 + 3: offset + p * 6 + 6]
        
        # Find active one-hot bit; if invalid (sum != 1), return sentinel -1
        primitives_shape.append(
            s_bits.index(1) if sum(s_bits) == 1 else -1
        )
        primitives_color.append(
            c_bits.index(1) if sum(c_bits) == 1 else -1
        )
    return primitives_shape, primitives_color


def _diff(lst):
    return len(set(lst)) == len(lst) and -1 not in lst


def _same(lst):
    return len(set(lst)) == 1 and -1 not in lst


def _pair(lst):
    return not _same(lst) and not _diff(lst) and -1 not in lst


def check(assignment) -> bool:
    """
    assignment[i] = 0 or 1 for DIMACS variable i+1.
    Returns True iff both images satisfy the shared Kand-Logic pattern.
    """
    a = list(assignment)

    shapes1, colors1 = _parse_image(a, 0)   # image 1: vars 1-18 (offset 0)
    shapes2, colors2 = _parse_image(a, 18)  # image 2: vars 19-36 (offset 18)

    # Invalid one-hot encoding → not a valid assignment for this problem
    if -1 in shapes1 or -1 in colors1 or -1 in shapes2 or -1 in colors2:
        return False

    # Per-image predicates
    sc1, dc1, pc1 = _same(colors1), _diff(colors1), _pair(colors1)
    sc2, dc2, pc2 = _same(colors2), _diff(colors2), _pair(colors2)
    ss1, ds1, ps1 = _same(shapes1), _diff(shapes1), _pair(shapes1)
    ss2, ds2, ps2 = _same(shapes2), _diff(shapes2), _pair(shapes2)

    return (
        (sc1 and sc2) or (pc1 and pc2) or (dc1 and dc2)
        or (ss1 and ss2) or (ps1 and ps2) or (ds1 and ds2)
    )
