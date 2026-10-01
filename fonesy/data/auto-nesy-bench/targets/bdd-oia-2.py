"""
BDD-OIA-2 (``bdd-oia-2``) oracle: move-forward/stop constraints.

This is a **multi-label oracle**: it takes the 9 *concept* variables as input
and returns 2 *action* output booleans (forward, stop). Simplified from the
full BDD-OIA oracle (bdd-oia.py) by dropping the
turn-left/turn-right concepts and actions entirely, keeping only the concepts
and rules needed to compute move_forward/stop_action.

Variables
---------
Concepts (vars 1-9 in the CNF, indices 0-8 here):
  Move-forward cues:   1=green_light  2=follow          3=road_clear
  Stop cues:           4=red_light    5=traffic_sign    6=car
                       7=person       8=rider           9=other_obstacle

Actions (vars 10-11 in the CNF, NOT sampled by the oracle):
  10=move_forward  11=stop_action

Reference: A Neuro-Symbolic Benchmark Suite for Concept Quality and
           Reasoning Shortcuts (Bortolotti et al.)
"""

N_VARS = 11           # total variables in the CNF (concepts + actions)
N_CONCEPT_VARS = 9    # variables the oracle samples over (concepts only)
N_CNF_VARS = 11        # alias — full CNF variable count

VARIABLE_NAMES = [
    "green_light", "follow", "road_clear",
    "red_light", "traffic_sign", "car", "person", "rider", "other_obstacle",
    "move_forward", "stop_action",
]

VARIABLE_DESCRIPTIONS = [
    "the traffic light ahead is green",
    "a lead vehicle is present and can be followed",
    "the road ahead is clear (no obstacles)",
    "the traffic light ahead is red",
    "a stop/yield sign is visible",
    "there is a car blocking the way",
    "there is a pedestrian crossing the street",
    "there is a rider blocking the way",
    "there is an obstacle blocking the way",
    "proceed straight ahead",
    "come to a stop",
]

ACTION_NAMES = ["move_forward", "stop_action"]

# Binary concept inputs have no structural one-hot or range constraints.
VALIDITY_SPEC = None


def check(concept_assignment: list) -> tuple:
    """Compute the expected 2 action values from the 9 concept inputs.

    Args:
        concept_assignment: A list of 9 int/bool values, one per concept
            variable (DIMACS vars 1-9, 0-indexed).

    Returns:
        A tuple ``(move_forward, stop_action)`` of 2 booleans representing
        the actions that the oracle logic mandates for this concept
        assignment.
    """
    (
        green_light, follow, road_clear,
        red_light, traffic_sign, car, person, rider, other_obstacle,
    ) = [bool(a) for a in concept_assignment]

    obstacle = car or person or rider or other_obstacle

    # move_forward: triggered by green_light, follow, or road_clear
    # but blocked when stop conditions hold
    can_forward = green_light or follow or road_clear
    must_stop = red_light or traffic_sign or obstacle
    move_forward = can_forward and not must_stop

    # stop_action: triggered by any stop condition
    stop_action = must_stop

    return (move_forward, stop_action)
