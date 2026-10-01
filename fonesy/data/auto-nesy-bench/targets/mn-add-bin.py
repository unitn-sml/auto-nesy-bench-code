"""
MNIST-Addition oracle (full single-digit range: digits 0-9, sum 0-18).

Binary encoding:
  Variables 1-4:  bits of digit 1  (d1_bit0 = LSB ... d1_bit3 = MSB)
  Variables 5-8:  bits of digit 2  (d2_bit0 = LSB ... d2_bit3 = MSB)
  Variables 9-13: bits of the sum  (sum_bit0 = LSB ... sum_bit4 = MSB)

  digit 1 = d1_bit3*8 + d1_bit2*4 + d1_bit1*2 + d1_bit0  (range 0-15, valid 0-9)
  digit 2 = d2_bit3*8 + d2_bit2*4 + d2_bit1*2 + d2_bit0  (range 0-15, valid 0-9)
  sum     = sum_bit4*16 + ... + sum_bit0                   (range 0-31, valid 0-18)

Multi-label interface: variables 1-8 are INPUTS (concept vars), variables 9-13
are OUTPUTS derived by the oracle (sum bits). N_CONCEPT_VARS separates them.
"""

N_VARS = 13
N_CONCEPT_VARS = 8   # input: 4 bits d1 + 4 bits d2
N_CNF_VARS = 13

VARIABLE_NAMES = [
    "d1_bit0", "d1_bit1", "d1_bit2", "d1_bit3",   # vars 1-4
    "d2_bit0", "d2_bit1", "d2_bit2", "d2_bit3",   # vars 5-8
    "sum_bit0", "sum_bit1", "sum_bit2", "sum_bit3", "sum_bit4",  # vars 9-13
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
    "the least significant bit of the sum",
    "the second bit of the sum",
    "the third bit of the sum",
    "the fourth bit of the sum",
    "the most significant bit of the sum",
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
    """Return the 5-bit sum encoding for the given 8-bit digit-pair input.

    This function is defined only for valid decimal-digit encodings. Evaluation
    must apply ``VALIDITY_SPEC`` first and treat an invalid concept assignment
    as having no correct output pattern.

    Args:
        concept_assignment: Sequence of 8 binary values encoding digits 0-9.
            Indices 0-3: bits of digit 1 (LSB first).
            Indices 4-7: bits of digit 2 (LSB first).

    Returns:
        5-tuple of int (0/1): binary encoding of d1+d2, LSB first.

    Raises:
        ValueError: If either 4-bit block encodes a value outside 0-9.
    """
    d1 = sum(concept_assignment[i] * (2**i) for i in range(4))
    d2 = sum(concept_assignment[4 + i] * (2**i) for i in range(4))
    if d1 > 9 or d2 > 9:
        raise ValueError(
            f"MNIST addition expects decimal digits in 0-9, got d1={d1}, d2={d2}"
        )
    s = d1 + d2
    return tuple((s >> i) & 1 for i in range(5))
