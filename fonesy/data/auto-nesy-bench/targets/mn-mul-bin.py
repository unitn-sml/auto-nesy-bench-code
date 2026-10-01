"""
MNIST-Multiplication oracle (full single-digit range: digits 0-9, product 0-81).
Binary encoding:
  Variables 1-4:   bits of digit 1   (d1_bit0 = LSB ... d1_bit3 = MSB)
  Variables 5-8:   bits of digit 2   (d2_bit0 = LSB ... d2_bit3 = MSB)
  Variables 9-15:  bits of the product (prod_bit0 = LSB ... prod_bit6 = MSB)
  digit 1 = d1_bit3*8 + d1_bit2*4 + d1_bit1*2 + d1_bit0       (range 0-15, valid 0-9)
  digit 2 = d2_bit3*8 + d2_bit2*4 + d2_bit1*2 + d2_bit0       (range 0-15, valid 0-9)
  product = prod_bit6*64 + prod_bit5*32 + ... + prod_bit0     (range 0-127, valid 0-81)
Multi-label interface: variables 1-8 are INPUTS (concept vars), variables 9-15
are OUTPUTS derived by the oracle (product bits). N_CONCEPT_VARS separates them.
"""
N_VARS = 15
N_CONCEPT_VARS = 8   # input: 4 bits d1 + 4 bits d2
N_CNF_VARS = 15
VARIABLE_NAMES = [
    "d1_bit0", "d1_bit1", "d1_bit2", "d1_bit3",   # vars 1-4
    "d2_bit0", "d2_bit1", "d2_bit2", "d2_bit3",   # vars 5-8
    "prod_bit0", "prod_bit1", "prod_bit2", "prod_bit3",
    "prod_bit4", "prod_bit5", "prod_bit6",        # vars 9-15
]

VARIABLE_DESCRIPTIONS = [
    "the least significant bit of the first digit",
    "the second bit of the first digit",
    "the third bit of the first digit",
    "the most significant bit of the first digit",
    "the least significant bit of the second digit",
    "the second bit of the second digit",
    "the third bit of the second digit",
    "the most significant bit of the second digit",
    "the least significant bit of the product",
    "the second bit of the product",
    "the third bit of the product",
    "the fourth bit of the product",
    "the fifth bit of the product",
    "the sixth bit of the product",
    "the most significant bit of the product",
]

# Concept variables: two 4-bit digits, each valid in [0, 9].
VALIDITY_SPEC = {
    "kind": "range_blocks",
    "blocks": [
        {"n_bits": 4, "max_val": 9},  # digit 1
        {"n_bits": 4, "max_val": 9},  # digit 2
    ],
}


def check(concept_assignment) -> tuple:
    """Return the 7-bit product encoding for the given 8-bit digit-pair input.

    This function is defined only for valid decimal-digit encodings. Evaluation
    must apply ``VALIDITY_SPEC`` first and treat an invalid concept assignment
    as having no correct output pattern.

    Args:
        concept_assignment: Sequence of 8 binary values encoding digits 0-9.
            Indices 0-3: bits of digit 1 (LSB first).
            Indices 4-7: bits of digit 2 (LSB first).

    Returns:
        7-tuple of int (0/1): binary encoding of d1*d2, LSB first.

    Raises:
        ValueError: If either 4-bit block encodes a value outside 0-9.
    """
    d1 = sum(concept_assignment[i] * (2**i) for i in range(4))
    d2 = sum(concept_assignment[4 + i] * (2**i) for i in range(4))
    if d1 > 9 or d2 > 9:
        raise ValueError(
            "MNIST multiplication expects decimal digits in 0-9, "
            f"got d1={d1}, d2={d2}"
        )
    p = d1 * d2
    return tuple((p >> i) & 1 for i in range(7))
