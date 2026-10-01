"""
ChestXRays (CHX) oracle: derive a 5-way severity code from 4 binary
radiological findings.

Multi-label interface: variables 1-4 are INPUTS (concept vars, one per
finding), variables 5-9 are OUTPUTS derived by the oracle (severity code,
one-hot). N_CONCEPT_VARS separates them.

Variable layout:
  Variables 1-4 (concepts): fracture, pneumothorax, airspace_opacity, nodule_mass
  Variables 5-9 (label, one-hot): healthy, green_code, yellow_code, orange_code, red_code

Let s = fracture + pneumothorax + airspace_opacity + nodule_mass (count of
true findings, 0-4). The severity code is exactly s:
  s=0 -> healthy   s=1 -> green_code   s=2 -> yellow_code
  s=3 -> orange_code   s=4 -> red_code
"""

N_VARS = 9
N_CONCEPT_VARS = 4   # input: 4 independent binary findings
N_CNF_VARS = 9

VARIABLE_NAMES = [
    "fracture", "pneumothorax", "airspace_opacity", "nodule_mass",  # vars 1-4
    "healthy", "green_code", "yellow_code", "orange_code", "red_code",  # vars 5-9
]

VARIABLE_DESCRIPTIONS = [
    "a bone fracture is visible",
    "a pneumothorax (collapsed lung) is visible",
    "airspace opacity (e.g. consolidation or infiltrate) is visible",
    "a nodule or mass is visible",
    "no findings are present",
    "green emergency code",
    "yellow emergency code",
    "orange emergency code",
    "red emergency code",
]

# Concept variables are independent binary findings — no one-hot/range
# structure to document (like bdd_oia's concepts).
VALIDITY_SPEC = None


def check(concept_assignment) -> tuple:
    """Return the one-hot 5-severity-level encoding for the given 4 findings.

    Args:
        concept_assignment: list of 4 int values (0/1) — fracture,
            pneumothorax, airspace_opacity, nodule_mass.

    Returns:
        5-tuple of bool, one-hot: (healthy, green_code, yellow_code,
        orange_code, red_code), true at index = count of true findings.
    """
    severity = sum(int(bool(c)) for c in concept_assignment[:4])
    label = [False] * 5
    label[severity] = True
    return tuple(label)
