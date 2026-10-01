"""
ROAD-R logical-requirements oracle (autonomous driving, single video frame).

Source: Giunchiglia et al., "ROAD-R: The Autonomous Driving Dataset with
Logical Requirements" (NeurIPS 2022 Datasets and Benchmarks Track).
https://github.com/EGiunchiglia/ROAD-R

Single-label interface: 41 Boolean variables encode which Agent/Action/
Location labels jointly hold for one detection in one video frame
  Variables 1-10 : Agent class (Pedestrian, Car, Cyclist, ...)   - one-hot
  Variables 11-13: Traffic-light colour (Red, Amber, Green)      - at most one
  Variables 14-29: Action (MovingAway, Braking, Crossing, ...)
  Variables 30-41: Location (VehicleLane, Junction, Parking, ...)
check(assignment) reports whether the full 41-variable assignment satisfies
every one of the 243 published logical requirements simultaneously.

Design note: unlike most other oracles in this package (sudoku, bdd_oia,
kand_logic, ...), road_r.dimacs is not *compiled* here from an independently
specified domain function - it already *is* the ground truth, published
verbatim by the ROAD-R authors. Hand-transcribing the 243 clauses into a
second, separately maintained Python boolean expression would only add a
second manual-transcription step with no independent value to cross-check
against. Instead, check() loads and evaluates the exact same road_r.dimacs
clauses shipped alongside this module, so the two representations can never
drift apart - by construction, not by testing. Genuinely independent
evidence of correctness instead comes from tests/test_road_r_oracle.py's
TestVariableMapping class, which hand-encodes a handful of structural facts
about the variable mapping and requirements (derived by cross-referencing
the official requirements_dimacs.txt against requirements_readable.txt) as
directly-constructed assignments.
"""

from pathlib import Path

from pysat.formula import CNF
from pysat.solvers import Solver

_DIMACS_PATH = Path(__file__).parent / "road-r.dimacs"

N_VARS = 41

VARIABLE_NAMES = [
    # Agent (1-10) - one-hot: exactly one is true
    "ped", "car", "cyc", "mobike", "med_veh", "lar_veh", "bus", "em_veh",
    "tl", "oth_tl",
    # Traffic-light colour (11-13) - at most one is true
    "red", "amber", "green",
    # Action (14-29)
    "mov_away", "mov_tow", "mov", "brake", "stop",
    "incat_left", "incat_rht", "haz_lit", "tur_lft", "tur_rht",
    "ovtak", "wait_2x", "xing_fm_lft", "xing_fm_rht", "xing", "push_obj",
    # Location (30-41)
    "veh_lane", "outgo_lane", "outgo_cyc_lane", "incom_lane",
    "incom_cyc_lane", "pav", "lft_pav", "rht_pav", "jun", "xing_loc",
    "bus_stop", "parking",
]

VARIABLE_DESCRIPTIONS = [
    "the detection is a pedestrian",
    "the detection is a car",
    "the detection is a cyclist",
    "the detection is a motorbike",
    "the detection is a medium vehicle",
    "the detection is a large vehicle",
    "the detection is a bus",
    "the detection is an emergency vehicle",
    "the detection is a traffic light",
    "the detection is another traffic light",
    "the traffic light is red",
    "the traffic light is amber",
    "the traffic light is green",
    "the detected agent is moving away",
    "the detected agent is moving towards the ego vehicle",
    "the detected agent is moving",
    "the detected agent is braking",
    "the detected agent is stopped",
    "the agent is indicating left",
    "the agent is indicating right",
    "the agent has its hazard lights on",
    "the agent is turning left",
    "the agent is turning right",
    "the agent is overtaking",
    "the agent is waiting to cross",
    "the agent is crossing from the left",
    "the agent is crossing from the right",
    "the agent is crossing",
    "the agent is pushing an object",
    "the detection is in a vehicle lane",
    "the detection is in an outgoing lane",
    "the detection is in an outgoing cycle lane",
    "the detection is in an incoming lane",
    "the detection is in an incoming cycle lane",
    "the detection is on the pavement",
    "the detection is on the left pavement",
    "the detection is on the right pavement",
    "the detection is at a junction",
    "the detection is at a crossing location",
    "the detection is at a bus stop",
    "the detection is in a parking area",
]

assert len(VARIABLE_NAMES) == N_VARS
assert len(VARIABLE_DESCRIPTIONS) == N_VARS

# The one-hot structure covers only the agent block.
VALIDITY_SPEC = None

_CLAUSES = None


def _get_clauses():
    """Load and cache the 243 clauses from the sibling road_r.dimacs file."""
    global _CLAUSES
    if _CLAUSES is None:
        _CLAUSES = CNF(from_file=str(_DIMACS_PATH)).clauses
    return _CLAUSES


def check(assignment: list) -> bool:
    """Return True iff the 41-variable assignment satisfies all 243 ROAD-R
    requirements simultaneously.

    Args:
        assignment: list of 41 ints/bools (0/1 or False/True), where
            assignment[k] is the truth value of DIMACS variable k+1.
    """
    a = list(assignment)
    assert len(a) == N_VARS, f"expected {N_VARS} values, got {len(a)}"
    for clause in _get_clauses():
        if not any((lit > 0) == bool(a[abs(lit) - 1]) for lit in clause):
            return False
    return True


def POSITIVE_SAMPLER(rng):
    """Return a random assignment satisfying all 243 requirements.

    Valid assignments are a tiny fraction of the 2**41 space (the Agent
    block alone is a strict one-hot over 10 variables), so plain rejection
    sampling essentially never finds one. Instead this biases a SAT solver
    towards a random point in its solution space: a fresh Solver instance
    plus a random +/- phase hint per variable (repeated Solver.solve() calls
    with no new hint are deterministic and would return the same model every
    time), then solve() - the CNF is trivially satisfiable (e.g. a
    stationary Car in the outgoing lane satisfies every requirement), so
    this always succeeds.
    """
    phases = [i if rng.random() < 0.5 else -i for i in range(1, N_VARS + 1)]
    with Solver(name="g3", bootstrap_with=_get_clauses()) as solver:
        solver.set_phases(phases)
        solver.solve()
        model = solver.get_model()
    return [1 if lit > 0 else 0 for lit in model[:N_VARS]]
