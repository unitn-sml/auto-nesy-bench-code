"""
CEBaB oracle: derive an overall review-sentiment label from 5 categorical
aspect-majority concepts (4 of them are the original CEBaB aspects; the 5th,
review_majority, was added on top so the concept-to-label mapping below is
deterministic).

Multi-label interface: variables 1-20 are INPUTS (5 concepts, one-hot
4-valued each), variables 21-25 are OUTPUTS derived by the oracle (label,
one-hot 5-valued). N_CONCEPT_VARS separates them.

Variable layout (each block one-hot, value order Positive/Negative/Neutral/unknown):
  Variables  1-4:  food_aspect_majority
  Variables  5-8:  service_aspect_majority
  Variables  9-12: noise_aspect_majority
  Variables 13-16: ambiance_aspect_majority
  Variables 17-20: review_majority (itself one of the 5 INPUTS, not the label)
  Variables 21-25 (label, one-hot): positive, negative, neutral, unknown, conflict

Let N_Pos, N_Neg, N_Neu, N_Unk be the count of the 5 concepts whose active
value is Positive/Negative/Neutral/unknown (each in 0-5). The label is:
  positive iff N_Pos > N_Neg and N_Pos >= N_Neu and N_Pos >= N_Unk
  negative iff N_Neg > N_Pos and N_Neg >= N_Neu and N_Neg >= N_Unk
  neutral  iff N_Neu > N_Pos and N_Neu > N_Neg and N_Neu >= N_Unk
  unknown  iff N_Unk > N_Pos and N_Unk > N_Neg and N_Unk > N_Neu
  conflict iff N_Pos == N_Neg and N_Pos >= N_Neu and N_Pos >= N_Unk
(Higher-priority categories use >= so they win ties; strictly lower-priority
categories use > so they lose ties; conflict is the Positive/Negative tie
at the overall max.)
"""

N_VARS = 25
N_CONCEPT_VARS = 20   # input: 5 concepts x 4-valued one-hot
N_CNF_VARS = 25

VARIABLE_NAMES = [
    "food_aspect_majority_positive", "food_aspect_majority_negative",
    "food_aspect_majority_neutral", "food_aspect_majority_unknown",
    "service_aspect_majority_positive", "service_aspect_majority_negative",
    "service_aspect_majority_neutral", "service_aspect_majority_unknown",
    "noise_aspect_majority_positive", "noise_aspect_majority_negative",
    "noise_aspect_majority_neutral", "noise_aspect_majority_unknown",
    "ambiance_aspect_majority_positive", "ambiance_aspect_majority_negative",
    "ambiance_aspect_majority_neutral", "ambiance_aspect_majority_unknown",
    "review_majority_positive", "review_majority_negative",
    "review_majority_neutral", "review_majority_unknown",
    "label_positive", "label_negative", "label_neutral",
    "label_unknown", "label_conflict",
]

VARIABLE_DESCRIPTIONS = [
    "positive sentiment toward the food",
    "negative sentiment toward the food",
    "neutral sentiment toward the food",
    "unknown sentiment toward the food",
    "positive sentiment toward the service",
    "negative sentiment toward the service",
    "neutral sentiment toward the service",
    "unknown sentiment toward the service",
    "positive sentiment toward the noise",
    "negative sentiment toward the noise",
    "neutral sentiment toward the noise",
    "unknown sentiment toward the noise",
    "positive sentiment toward the ambiance",
    "negative sentiment toward the ambiance",
    "neutral sentiment toward the ambiance",
    "unknown sentiment toward the ambiance",
    "positive overall review sentiment",
    "negative overall review sentiment",
    "neutral overall review sentiment",
    "unknown overall review sentiment",
    "final assessment of the restaurant is positive",
    "final assessment of the restaurant is negative",
    "final assessment of the restaurant is neutral",
    "final assessment of the restaurant is unknown",
    "final assessment of the restaurant is conflicting",
]

# 5 concepts, each a 4-way one-hot block (Positive/Negative/Neutral/unknown).
VALIDITY_SPEC = {
    "kind": "one_hot_blocks",
    "block_layout": [4, 4, 4, 4, 4],
}


def _block_index(bits) -> int:
    """Decode a 4-bit one-hot block to its active index, or -1 if malformed."""
    return bits.index(1) if sum(bits) == 1 else -1


def check(concept_assignment) -> tuple:
    """Return the one-hot 5-label encoding for the given 20-bit concept input.

    Args:
        concept_assignment: list of 20 int values (0/1) — 5 one-hot 4-valued
            blocks (Positive/Negative/Neutral/unknown), in the order food,
            service, noise, ambiance, review_majority.

    Returns:
        5-tuple of bool, one-hot: (positive, negative, neutral, unknown,
        conflict). All-False if any concept block isn't a valid one-hot.
    """
    a = list(concept_assignment)
    idxs = [_block_index(a[4 * k: 4 * k + 4]) for k in range(5)]
    if -1 in idxs:
        return (False, False, False, False, False)

    n_pos = idxs.count(0)
    n_neg = idxs.count(1)
    n_neu = idxs.count(2)
    n_unk = idxs.count(3)

    if n_pos > n_neg and n_pos >= n_neu and n_pos >= n_unk:
        label = 0  # positive
    elif n_neg > n_pos and n_neg >= n_neu and n_neg >= n_unk:
        label = 1  # negative
    elif n_neu > n_pos and n_neu > n_neg and n_neu >= n_unk:
        label = 2  # neutral
    elif n_unk > n_pos and n_unk > n_neg and n_unk > n_neu:
        label = 3  # unknown
    elif n_pos == n_neg and n_pos >= n_neu and n_pos >= n_unk:
        label = 4  # conflict
    else:
        # Unreachable: the 5 branches above are exhaustive over all counts
        # (verified by brute force over all 4^5 combos in
        # tests/test_cebab_oracle.py), but fail loudly rather than silently
        # mislabeling if that invariant is ever broken.
        raise AssertionError(f"No label branch matched counts {(n_pos, n_neg, n_neu, n_unk)}")

    result = [False] * 5
    result[label] = True
    return tuple(result)
