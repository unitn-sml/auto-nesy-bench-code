"""Convert DCP-Bench-Open constraint problems into the ``dcp-bench`` benchmark.

For every problem of ``fonesy/data/dcp-bench/raw/dcp-bench-open.jsonl``, the
reference CPMpy model is instantiated with its example instance and compiled
to CNF with CPMpy's ``to_cnf`` (one-hot, "direct" encoding). Optimization
problems, conversions exceeding ``--timeout`` seconds and CNFs not satisfied
by the provided example solution are skipped. Writes::

    fonesy/data/dcp-bench/inputs/<id>.txt       problem description
    fonesy/data/dcp-bench/targets/<id>.dimacs   ground-truth CNF
    fonesy/data/dcp-bench/targets/<id>.py       VARIABLE_NAMES, N_PRIMARY_VARS, DECISION_VARIABLES
    fonesy/data/dcp-bench/targets/<id>_model.py instantiated CPMpy model

Usage::

    python ground-truth-dimacs/dcp-bench.py [--output-dir DIR] [--timeout SECONDS]
"""

import argparse
import json
import re
import sys
import textwrap
import traceback
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cpmpy as cp
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.variables import NegBoolView, _BoolVarImpl, _IntVarImpl
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.to_cnf import to_cnf

from fonesy.evaluation.solver import _run_hard_timeout
from fonesy.datasets.prompts import format_variable_list_grouped

RAW_FILE = _ROOT / "fonesy" / "data" / "dcp-bench" / "raw" / "dcp-bench-open.jsonl"
DEFAULT_OUT = _ROOT / "fonesy" / "data" / "dcp-bench"
DEFAULT_TIMEOUT = 90  # seconds per problem

# Drop the trailing "Print ..." sentence of the original (execute-and-print) task.
_TRAILING_PRINT_INSTRUCTION_RE = re.compile(r"\s*\bPrint\b.*$", re.DOTALL)

_INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9_]+")
_ARRAY_ELEMENT_RE = re.compile(r"^(\w+)\[[\d,]+\]$")

# {variables} lists the CP decision variables; auxiliary variables (e.g. from
# flattening) are never declared, the model may introduce its own.
_PROBLEM_TEMPLATE = """This problem is derived from a constraint programming (CP) model.{instance_data}

{problem}

Use the mapped variables in the provided order; you can introduce auxiliary variables."""


def _format_instance_data(instance_code: str) -> str:
    """Render *instance_code* (the raw ``example_instance`` source) into a
    prompt section, or ``""`` when there is none. Shown verbatim: many
    problems' variable counts/shapes are only fixed by this data, and the
    puzzle description alone never restates it.
    """
    instance_code = instance_code.strip()
    if not instance_code:
        return ""
    return (
        "\n\nThis problem's concrete instance data (from the constraint programming model) is:\n\n"
        f"{instance_code}\n\n"
        "This instance data determines the specific constraints of the problem below and MUST be "
        "correctly encoded into the CNF clauses you produce -- it is not optional background; the "
        "puzzle description that follows only explains the general rules, not this instance's "
        "specific values."
    )


def _sanitize_identifier(text: str) -> str:
    """Collapse *text* into a legal, non-empty Python-identifier-shaped string."""
    candidate = _INVALID_CHARS_RE.sub("_", text).strip("_")
    if not candidate:
        candidate = "v"
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def _disambiguate(candidate: str, seen: set) -> str:
    """Return *candidate*, or a numerically-suffixed variant, guaranteed
    unique against *seen* (which is updated in place)."""
    unique = candidate
    suffix = 1
    while unique in seen:
        unique = f"{candidate}_{suffix}"
        suffix += 1
    seen.add(unique)
    return unique


def _capture_model(code: str, instance_code: str = ""):
    """Execute CPMpy problem code without solving; return (Model, namespace).

    Instance code (parameterised problems' concrete values, e.g. ``n``) is
    exec'd first so ``code`` can reference it.
    """
    original_solve = cp.Model.solve
    cp.Model.solve = lambda self, **_: True

    ns: dict = {}
    exec("from cpmpy import *\nimport json\nimport math", ns)

    if instance_code.strip():
        try:
            exec(instance_code, ns)
        except Exception:
            pass  # best-effort; model code may still succeed without it

    try:
        exec(code, ns)
    finally:
        cp.Model.solve = original_solve

    return ns.get("model"), ns


def _build_ns_maps(ns: dict) -> tuple[dict, dict]:
    """Scan the exec namespace *ns* for variable bindings, returning
    ``(id_to_name, id_to_index)``:

    - ``id_to_name``: ``id(variable) -> its Python identifier``. Needed
      because several problems never pass ``name=`` to ``boolvar()``/
      ``intvar()``, so cpmpy auto-generates internal names ("BV0", ...)
      indistinguishable from genuine Tseitin auxiliaries; the source name is
      always meaningful.
    - ``id_to_index``: ``id(variable) -> its position within the array it
      was bound to``. Needed so array-element name suffixes reflect the
      *true* array index rather than constraint-discovery order, which can
      disagree (e.g. a model states ``arr[-1] == y`` before looping over
      the array, so the element referenced second isn't index 1).

    ``setdefault`` (not overwrite) in both maps so the first/array-level
    binding wins over a later scalar alias to the same object (e.g.
    ``half_crowns = coins[0]``).
    """
    id_to_name: dict = {}
    id_to_index: dict = {}
    for key, val in ns.items():
        if key.startswith("_"):
            continue
        if isinstance(val, _IntVarImpl):  # covers _BoolVarImpl too
            id_to_name.setdefault(id(val), key)
        elif hasattr(val, "flat"):  # NDVarArray-like
            try:
                for i, elem in enumerate(val.flat):
                    if isinstance(elem, _IntVarImpl):
                        id_to_name.setdefault(id(elem), key)
                        id_to_index.setdefault(id(elem), i)
            except TypeError:
                continue
    return id_to_name, id_to_index


def _describe_decision_variables(variable_names: list) -> str:
    """Render the primary variables as ranges with domains, e.g.
    ``1-101: men: integer, domain [0, 100]``, from the compiled one-hot bit
    names (``to_cnf`` may drop some bits, so domains are not taken from the
    model). Auxiliary variables are never listed."""
    return format_variable_list_grouped(variable_names)


def _primary_variable_names(model: cp.Model, ns: dict) -> list[str]:
    """Friendly, unique names for *model*'s pre-CNF decision variables, in
    ``_describe_decision_variables``'s presentation order. Prefers the
    source Python identifier; array elements sharing a base get their true
    array-index suffix (``statements_0``, ...) from ``_build_ns_maps``,
    falling back to a discovery-order counter or synthesized ``var<i>`` when
    no source name/index is recoverable. Feeds the naming block appended to
    each problem's materialized ``<id>_model.py`` (``_build_named_model_text``).
    """
    variables = get_variables(toplevel_list(model.constraints))
    id_to_name, id_to_index = _build_ns_maps(ns)
    friendly = [id_to_name.get(id(v)) or f"var{i}" for i, v in enumerate(variables)]

    counts = Counter(friendly)
    next_index: dict = {}
    seen: set = set()
    names = []
    for v, base in zip(variables, friendly):
        if counts[base] <= 1:
            names.append(_disambiguate(base, seen))
            continue
        idx = id_to_index.get(id(v))
        if idx is None:
            idx = next_index.get(base, 0)
            next_index[base] = idx + 1
        names.append(_disambiguate(f"{base}_{idx}", seen))
    return names


_MODEL_FILE_TEMPLATE = '''"""Ground-truth CPMpy model for {pid!r}.
Compiled to fonesy/data/dcp-bench/targets/{pid}.dimacs.
Generated by data/dcp-bench/convert_to_cnf.py -- do not edit by hand."""

{instance_code}
{code}
# --- primary variable naming, in the same order shown in the prompt ---
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

_primary_variable_names = {names!r}
for _v, _n in zip(get_variables(toplevel_list(model.constraints)), _primary_variable_names):
    _v.name = _n
'''


def _build_named_model_text(pid: str, code: str, instance_code: str, names: list) -> str:
    """Render ``<id>_model.py``: the original model code plus a naming block
    giving every primary variable a real name. The CNF is compiled from
    *this* text, so the file on disk always matches the ``.dimacs``.
    """
    return _MODEL_FILE_TEMPLATE.format(
        pid=pid, instance_code=instance_code.strip(), code=code.strip(), names=names
    )


def _find_defining_expr(var, flat_constraints: list):
    """Search *flat_constraints* for a clean ``var == expr`` (or
    ``expr == var``) equation defining *var*; ``None`` if none is found.
    """
    for c in flat_constraints:
        if isinstance(c, Comparison) and c.name == "==":
            lhs, rhs = c.args
            if lhs is var:
                return rhs
            if rhs is var:
                return lhs
    return None


def _discover_extra_primary_variables(flat_constraints: list, pre_var_ids: set):
    """Find variables cpmpy's own constraint *flattening* introduces beyond
    the model's declared decision variables (e.g. an intermediate for
    ``abs(green - white)``), name each one (``aux0``, ``aux1``, ...) and
    render a human-readable description from its defining equation (see
    ``_find_defining_expr``). Names are assigned in a first pass so
    cross-references between extras render correctly regardless of
    discovery order.

    Returns ``(extra_vars, descriptions)`` -- ``extra_vars`` in discovery
    order, ``descriptions`` as parallel ``(name, rendered_description)`` pairs.
    """
    seen_ids: set = set()
    extra_vars: list = []
    for c in flat_constraints:
        for v in get_variables([c]):
            if id(v) not in pre_var_ids and id(v) not in seen_ids:
                seen_ids.add(id(v))
                extra_vars.append(v)

    for i, v in enumerate(extra_vars):
        v.name = f"aux{i}"

    descriptions = []
    for v in extra_vars:
        expr = _find_defining_expr(v, flat_constraints)
        domain_suffix = "" if isinstance(v, _BoolVarImpl) else f" (domain [{v.lb}, {v.ub}])"
        if expr is not None:
            descriptions.append((v.name, f"{v.name} = {expr}{domain_suffix}"))
        else:
            descriptions.append((
                v.name,
                f"{v.name}: auxiliary quantity introduced by the constraints above{domain_suffix}",
            ))
    return extra_vars, descriptions


def _to_cnf_and_names(
    model: cp.Model, no_aux_vars: bool = False
) -> tuple[str, list[str], int, list]:
    """Compile *model* to DIMACS with CPMpy's ``to_cnf`` (one-hot "direct" encoding).

    Returns ``(dimacs_str, variable_names, primary_num_vars, extra_descriptions)``
    with the primary variables first. Primary variables are the declared decision
    variables (plus, unless *no_aux_vars*, the variables introduced by
    flattening), matched by object identity; every other CNF variable is an
    auxiliary one. *model* must come from a materialized ``<id>_model.py``.
    """
    constraints = toplevel_list(model.constraints)
    pre_vars = get_variables(constraints)
    pre_var_ids = {id(v) for v in pre_vars}

    flat = flatten_constraint(constraints)
    if no_aux_vars:
        extra_vars, extra_descriptions = [], []
    else:
        extra_vars, extra_descriptions = _discover_extra_primary_variables(flat, pre_var_ids)
    all_primary = list(pre_vars) + extra_vars

    ivarmap: dict = {}
    cnf = to_cnf(flat, ivarmap=ivarmap, encoding="direct")
    cnf_vars = get_variables(cnf)
    cnf_var_id_set = {id(v) for v in cnf_vars}

    # ivarmap is keyed by cpmpy's own internal name, not ours, and not
    # reliably recoverable post-to_cnf() -- resolve each encoder back to the
    # variable object it encodes (enc._x) and index by identity instead.
    encoder_by_var_id = {id(enc._x): enc for enc in ivarmap.values()}

    seen: set = set()
    primary_vars: list = []
    variable_names: list = []
    for v in all_primary:
        if id(v) in encoder_by_var_id:
            enc = encoder_by_var_id[id(v)]
            lb = enc._x.lb
            for offset, bit in enumerate(enc.vars()):
                if id(bit) not in cnf_var_id_set:
                    continue  # eliminated by cpmpy's own simplification
                primary_vars.append(bit)
                value = lb + offset
                value_str = f"m{-value}" if value < 0 else str(value)
                candidate = _sanitize_identifier(f"{v.name}_eq{value_str}")
                variable_names.append(_disambiguate(candidate, seen))
        elif id(v) in cnf_var_id_set:
            primary_vars.append(v)
            variable_names.append(_disambiguate(_sanitize_identifier(v.name), seen))
        # else: fully eliminated by cpmpy's own simplification -- nothing to emit.

    primary_id_set = {id(v) for v in primary_vars}
    auxiliary_vars = [v for v in cnf_vars if id(v) not in primary_id_set]
    for i in range(len(auxiliary_vars)):
        variable_names.append(_disambiguate(f"aux_bv{i}", seen))

    vars_ = primary_vars + auxiliary_vars
    primary_num_vars = len(primary_vars)
    mapping = {v: i + 1 for i, v in enumerate(vars_)}

    lines = [f"p cnf {len(vars_)} {len(cnf)}"]
    for cons in cnf:
        if isinstance(cons, _BoolVarImpl):
            cons = Operator("or", [cons])
        if not (isinstance(cons, Operator) and cons.name == "or"):
            raise NotImplementedError(f"Unsupported constraint {cons}")

        ints = []
        for v in cons.args:
            if isinstance(v, NegBoolView):
                ints.append(str(-mapping[v._bv]))
            elif isinstance(v, _BoolVarImpl):
                ints.append(str(mapping[v]))
            else:
                raise ValueError(f"Expected Boolean variable in clause, but got {v!r}")
        lines.append(" ".join(ints + ["0"]))

    return "\n".join(lines) + "\n", variable_names, primary_num_vars, extra_descriptions


def _assign_value(var, value) -> None:
    """Recursively assign *value* into *var*, descending into nested
    array/grid variables (a plain ``var._value = value`` only sticks for
    scalars)."""
    if isinstance(value, (list, tuple)):
        for sub_var, sub_val in zip(var, value):
            _assign_value(sub_var, sub_val)
    else:
        var._value = value


def _all_true(value) -> bool:
    """``bool()`` that also handles array/nested-list-shaped constraint
    values (some CPMpy constraints evaluate to a numpy array of booleans)."""
    if isinstance(value, (list, tuple)):
        return all(_all_true(v) for v in value)
    if hasattr(value, "flat"):
        return all(bool(v) for v in value.flat)
    return bool(value)


def _check_example_solution(ns: dict, model: cp.Model, example_solution: dict) -> bool:
    """Return whether *example_solution* extends to a full satisfying
    assignment of *model*.

    Fast path: assign the named variables and evaluate every constraint
    directly. Falls back to re-solving with only the named variables fixed
    (via z3, not the default backend, which can't drive some of these
    models) when the model has unset auxiliary variables the fast path
    can't resolve.
    """
    if not example_solution:
        return False

    named_vars = []
    for name, value in example_solution.items():
        var = ns.get(name)
        if var is None:
            return False
        named_vars.append((var, value))

    try:
        for var, value in named_vars:
            _assign_value(var, value)
        constraints = toplevel_list(model.constraints)
        if all(_all_true(c.value()) for c in constraints):
            return True
    except Exception:
        pass  # fall through to the slow path (e.g. unset auxiliary variables)

    fixed_constraints = []
    for var, value in named_vars:
        try:
            fixed_constraints.append(var == value)
        except Exception:
            return False
    try:
        check_model = cp.Model(list(model.constraints) + fixed_constraints)
        return bool(check_model.solve(solver="z3"))
    except Exception:
        return False


def _write(path: Path, text: str, dry_run: bool):
    if dry_run:
        print(f"  [DRY-RUN] would write {path} ({len(text)} bytes)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _convert_one(
    pid: str,
    desc: str,
    code: str,
    instance_code: str,
    example_solution: dict,
    inputs_dir: Path,
    targets_dir: Path,
    dry_run: bool,
) -> dict:
    """Convert one problem and write its files; return a small summary dict.

    Runs in a hard-timeout subprocess (``to_cnf`` can hang); files are written
    here because returning multi-MB strings through the result queue can
    deadlock. Variables introduced by flattening are auxiliary
    (``no_aux_vars=True``).
    """
    model, ns = _capture_model(code, instance_code)
    # `from cpmpy import *` also binds the bare name `model` to the
    # `cpmpy.model` submodule, so check the type explicitly rather than `is None`.
    if not isinstance(model, cp.Model):
        return {"status": "fail", "reason": "no 'model' variable found after exec()"}
    if not model.constraints:
        return {"status": "fail", "reason": "model has no constraints"}
    if model.has_objective():
        return {"status": "optimization"}

    if not _check_example_solution(ns, model, example_solution):
        return {"status": "fail", "reason": "example_solution does not extend to a solution of the model"}

    # Compile CNF from the *materialized* named model (not the original
    # code) so the saved file always matches the .dimacs.
    names = _primary_variable_names(model, ns)
    model_text = _build_named_model_text(pid, code, instance_code, names)
    named_model, _named_ns = _capture_model(model_text, "")
    if not isinstance(named_model, cp.Model):
        return {"status": "fail", "reason": "materialized named model has no 'model' variable"}

    dimacs_str, variable_names, primary_num_vars, _extra_descriptions = _to_cnf_and_names(
        named_model, no_aux_vars=True
    )

    header = dimacs_str.strip().splitlines()[0]
    n_vars, n_clauses = header.split()[2], header.split()[3]

    problem_text = _TRAILING_PRINT_INSTRUCTION_RE.sub("", desc).strip()
    decision_variables = _describe_decision_variables(variable_names[:primary_num_vars])
    enriched_desc = _PROBLEM_TEMPLATE.format(
        instance_data=_format_instance_data(instance_code),
        problem=problem_text,
    )
    _write(inputs_dir / f"{pid}.txt", enriched_desc, dry_run)
    _write(targets_dir / f"{pid}.dimacs", dimacs_str, dry_run)
    _write(targets_dir / f"{pid}_model.py", model_text, dry_run)
    _write(
        targets_dir / f"{pid}.py",
        f"VARIABLE_NAMES = {variable_names!r}\n"
        f"N_PRIMARY_VARS = {primary_num_vars!r}\n"
        f"DECISION_VARIABLES = {decision_variables!r}\n",
        dry_run,
    )

    return {"status": "ok", "n_vars": n_vars, "n_clauses": n_clauses}


def main():
    parser = argparse.ArgumentParser(description="Convert DCP-Bench-Open to CNF DIMACS")
    parser.add_argument("--dry-run",    action="store_true", help="Print actions without writing")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                         help="Per-problem hard timeout in seconds (default: %(default)s)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    inputs_dir = out_dir / "inputs"
    targets_dir = out_dir / "targets"
    dry_run = args.dry_run

    if not RAW_FILE.exists():
        sys.exit(f"Raw file not found: {RAW_FILE}\nRun: make dcp-bench")

    problems = [json.loads(l) for l in RAW_FILE.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(problems)} problems from {RAW_FILE}")
    print(f"Output dir: {out_dir}")
    print()

    ok_ids: list[str] = []
    optimization_ids: list[str] = []
    fail_ids: list[tuple[str, str]] = []
    timeout_ids: list[str] = []

    for prob in problems:
        pid = prob["id"]
        desc = prob.get("description", "").strip()
        code = prob.get("model", "")
        instance_code = prob.get("example_instance", "")
        example_solution = prob.get("example_solution") or {}

        print(f"  {pid} ... ", end="", flush=True)

        try:
            result = _run_hard_timeout(
                _convert_one,
                (pid, desc, code, instance_code, example_solution, inputs_dir, targets_dir, dry_run),
                args.timeout, pid,
            )
        except TimeoutError:
            print(f"TIMEOUT (> {args.timeout}s)")
            timeout_ids.append(pid)
            continue
        except RuntimeError as exc:
            msg = textwrap.shorten(str(exc), width=100)
            print(f"FAIL  — {msg}")
            fail_ids.append((pid, str(exc)))
            continue
        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            print(f"FAIL  — {textwrap.shorten(msg, width=100)}")
            fail_ids.append((pid, msg))
            continue

        if result["status"] == "optimization":
            print("SKIP  — optimization problem")
            optimization_ids.append(pid)
            continue
        if result["status"] == "fail":
            print(f"FAIL  — {result['reason']}")
            fail_ids.append((pid, result["reason"]))
            continue

        print(f"OK  ({result['n_vars']} vars, {result['n_clauses']} clauses)")
        ok_ids.append(pid)

    print()
    print(f"Converted: {len(ok_ids)} / {len(problems)}")
    print(f"Excluded (optimization problems): {len(optimization_ids)}")
    if fail_ids:
        print(f"Failed ({len(fail_ids)}):")
        for fid, reason in fail_ids:
            print(f"  {fid}: {textwrap.shorten(reason, 100)}")
    if timeout_ids:
        print(f"Timed out (> {args.timeout}s) ({len(timeout_ids)}): {', '.join(timeout_ids)}")
    print()

    if not dry_run:
        _write(
            out_dir / "excluded_optimization.json",
            json.dumps(sorted(optimization_ids), indent=2) + "\n",
            dry_run=False,
        )


if __name__ == "__main__":
    main()
