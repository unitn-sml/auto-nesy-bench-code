"""
BDD-OIA oracle for road-driving action constraints.

This is a **multi-label oracle**: it takes the 21 *concept* variables as input
and returns 4 *action* output booleans (forward, stop, left, right).  The
evaluator detects multi-label oracles via the presence of ``N_CONCEPT_VARS``.

Variables
---------
Concepts (vars 1-21 in the CNF, indices 0-20 here):
  Move-forward cues:   1=green_light  2=follow          3=road_clear
  Stop cues:           4=red_light    5=traffic_sign    6=car
                       7=person       8=rider           9=other_obstacle
  Turn-left cues:     10=left_lane   11=left_green_light 12=left_follow
                      13=no_left_lane 14=left_obstacle  15=left_solid_line
  Turn-right cues:    16=right_lane  17=right_green_light 18=right_follow
                      19=no_right_lane 20=right_obstacle 21=right_solid_line

Actions (vars 22-25 in the CNF, NOT sampled by the oracle):
  22=move_forward  23=stop_action  24=turn_left  25=turn_right

Reference: A Neuro-Symbolic Benchmark Suite for Concept Quality and
           Reasoning Shortcuts (Bortolotti et al.)
"""

N_VARS = 25           # total variables in the CNF (concepts + actions)
N_CONCEPT_VARS = 21   # variables the oracle samples over (concepts only)
N_CNF_VARS = 25       # alias — full CNF variable count

VARIABLE_NAMES = [
    "green_light", "follow", "road_clear",
    "red_light", "traffic_sign", "car", "person", "rider", "other_obstacle",
    "left_lane", "left_green_light", "left_follow",
    "no_left_lane", "left_obstacle", "left_solid_line",
    "right_lane", "right_green_light", "right_follow",
    "no_right_lane", "right_obstacle", "right_solid_line",
    "move_forward", "stop_action", "turn_left", "turn_right",
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
    "there is a left lane enabling a left turn",
    "there is a green light enabling a left turn",
    "a lead vehicle on the left lane is present and can be followed",
    "there is no left lane",
    "there is an obstacle blocking the left lane",
    "there is a solid line preventing a left turn",
    "there is a right lane enabling a left turn",
    "there is a green light enabling a right turn",
    "a lead vehicle on the right lane is present and can be followed",
    "there is no right lane",
    "there is an obstacle blocking the right lane",
    "there is a solid line preventing a right turn",
    "proceed straight ahead",
    "come to a stop",
    "turn left",
    "turn right",
]

ACTION_NAMES = ["move_forward", "stop_action", "turn_left", "turn_right"]

# Binary concept inputs have no structural one-hot or range constraints.
VALIDITY_SPEC = None


def check(concept_assignment: list) -> tuple:
    """Compute the expected 4 action values from the 21 concept inputs.

    Args:
        concept_assignment: A list of 21 int/bool values, one per concept
            variable (DIMACS vars 1-21, 0-indexed).

    Returns:
        A tuple ``(move_forward, stop_action, turn_left, turn_right)`` of
        4 booleans representing the actions that the oracle logic mandates
        for this concept assignment.
    """
    (
        green_light, follow, road_clear,
        red_light, traffic_sign, car, person, rider, other_obstacle,
        left_lane, left_green_light, left_follow,
        no_left_lane, left_obstacle, left_solid_line,
        right_lane, right_green_light, right_follow,
        no_right_lane, right_obstacle, right_solid_line,
    ) = [bool(a) for a in concept_assignment]

    obstacle = car or person or rider or other_obstacle

    # move_forward: triggered by green_light, follow, or road_clear
    # but blocked when stop conditions hold
    can_forward = green_light or follow or road_clear
    must_stop = red_light or traffic_sign or obstacle
    move_forward = can_forward and not must_stop

    # stop_action: triggered by any stop condition
    stop_action = must_stop

    # turn_left: can turn and no blocking condition
    can_turn_left = left_lane or left_green_light or left_follow
    cant_turn_left = no_left_lane or left_obstacle or left_solid_line
    turn_left = can_turn_left and not cant_turn_left

    # turn_right: symmetric to left
    can_turn_right = right_lane or right_green_light or right_follow
    cant_turn_right = no_right_lane or right_obstacle or right_solid_line
    turn_right = can_turn_right and not cant_turn_right

    return (move_forward, stop_action, turn_left, turn_right)
