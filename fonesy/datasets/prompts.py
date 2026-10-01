"""Prompt templates and per-problem variable metadata shared by all datasets."""

import importlib.util
import re
from collections.abc import Sequence
from pathlib import Path

from fonesy.datasets.base import VariableSpec
from fonesy.general_utils import load_file

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DATA_DIR = Path(__file__).parent.parent / "data"

MODE_CHOICES = ["zero-shot", "zero-shot-cot"]
CNF_FORMATS = ["dimacs", "nat", "pysat", "cpmpy", "sympy"]


def load_prompt_template(benchmark: str, cnf_format: str, mode: str) -> str:
    """Return the prompt template ``<benchmark>_<format>[_cot].txt``."""
    if mode not in MODE_CHOICES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {MODE_CHOICES}")
    if cnf_format not in CNF_FORMATS:
        raise ValueError(f"Unknown cnf_format '{cnf_format}'. Choose from {CNF_FORMATS}")
    suffix = "_cot" if mode == "zero-shot-cot" else ""
    return load_file(PROMPTS_DIR / f"{benchmark}_{cnf_format}{suffix}.txt")


def load_module(path):
    """Import a Python file (oracle / variable metadata) as a module."""
    spec = importlib.util.spec_from_file_location("oracle", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_variable_list(variable_names, descriptions=None):
    """Render one ``<idx>: <name>[: <description>]`` line per variable."""
    if descriptions is None:
        return "\n".join(f"  {i + 1}: {name}" for i, name in enumerate(variable_names))
    return "\n".join(
        f"  {i + 1}: {name}: {description}"
        for i, (name, description) in enumerate(zip(variable_names, descriptions))
    )


_ONE_HOT_EQ_SUFFIX_RE = re.compile(r"^(.*)_eq(m?\d+)$")


def format_variable_list_grouped(variable_names):
    """Collapse runs of one-hot bits ``<name>_eq<v>`` into one line per variable.

    For example ``green_eq0 .. green_eq4`` becomes
    ``1-5: green: integer, domain [0, 4]``. Other names get a single line.
    """
    lines = []
    i, n = 0, len(variable_names)
    while i < n:
        match = _ONE_HOT_EQ_SUFFIX_RE.match(variable_names[i])
        if match is None:
            lines.append(f"  {i + 1}: {variable_names[i]}")
            i += 1
            continue
        base = match.group(1)
        start = i
        values = []
        while i < n:
            m = _ONE_HOT_EQ_SUFFIX_RE.match(variable_names[i])
            if m is None or m.group(1) != base:
                break
            val_str = m.group(2)
            values.append(-int(val_str[1:]) if val_str.startswith("m") else int(val_str))
            i += 1
        index_range = f"{start + 1}" if i - 1 == start else f"{start + 1}-{i}"
        lines.append(f"  {index_range}: {base}: integer, domain [{min(values)}, {max(values)}]")
    return "\n".join(lines)


def render_prompt(prompt_template, problem_text, variables_text):
    """Fill the ``{variables}`` and ``{problem}`` placeholders of a template."""
    if "{variables}" not in prompt_template:
        raise ValueError("Prompt template is missing the {variables} placeholder")
    if "{problem}" not in prompt_template:
        raise ValueError("Prompt template is missing the {problem} placeholder")
    return prompt_template.replace("{variables}", variables_text).replace(
        "{problem}", problem_text
    )


def dimacs_num_variables(target_text, problem_name):
    """Return the variable count declared in a DIMACS header."""
    for line in target_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[:2] == ["p", "cnf"]:
            try:
                return int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid DIMACS variable count for '{problem_name}': {parts[2]!r}"
                ) from exc
    raise ValueError(f"Missing DIMACS header for '{problem_name}'")


def _validate_variable_names(variable_names, problem_name, target_text, declared_num_vars=None):
    if not isinstance(variable_names, Sequence) or isinstance(variable_names, (str, bytes)):
        raise ValueError(f"VARIABLE_NAMES for '{problem_name}' must be a sequence of strings")
    names = list(variable_names)
    if not names:
        raise ValueError(f"VARIABLE_NAMES for '{problem_name}' must not be empty")
    for index, name in enumerate(names, start=1):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"VARIABLE_NAMES for '{problem_name}' contains an invalid name "
                f"at position {index}: {name!r}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"VARIABLE_NAMES for '{problem_name}' contains duplicates")

    num_variables = dimacs_num_variables(target_text, problem_name)
    if len(names) != num_variables:
        raise ValueError(
            f"VARIABLE_NAMES for '{problem_name}' has {len(names)} entries, "
            f"but its DIMACS target declares {num_variables} variables"
        )
    if declared_num_vars is not None and declared_num_vars != num_variables:
        raise ValueError(
            f"N_VARS for '{problem_name}' is {declared_num_vars}, "
            f"but its DIMACS target declares {num_variables} variables"
        )
    return names


def _validate_primary_num_vars(value, problem_name, num_variables):
    """Validate an optional ``N_PRIMARY_VARS`` (number of leading non-auxiliary variables)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"N_PRIMARY_VARS for '{problem_name}' must be an integer")
    if not (1 <= value <= num_variables):
        raise ValueError(
            f"N_PRIMARY_VARS for '{problem_name}' must be between 1 and "
            f"{num_variables} (got {value})"
        )
    return value


def load_variable_spec(metadata_file, problem_name, target_text):
    """Build the :class:`VariableSpec` of a problem from its ``<id>.py`` metadata.

    The module must define ``VARIABLE_NAMES`` (one per DIMACS variable of the
    target) and may define ``N_VARS`` and ``N_PRIMARY_VARS``.
    """
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing variable metadata for '{problem_name}': {metadata_file}")
    module = load_module(metadata_file)
    if not hasattr(module, "VARIABLE_NAMES"):
        raise ValueError(f"'{metadata_file}' must define VARIABLE_NAMES")
    names = _validate_variable_names(
        module.VARIABLE_NAMES, problem_name, target_text, getattr(module, "N_VARS", None)
    )
    primary_num_vars = _validate_primary_num_vars(
        getattr(module, "N_PRIMARY_VARS", None), problem_name, len(names)
    )
    return VariableSpec.from_source_names(names, primary_num_vars=primary_num_vars)


def load_variable_descriptions(oracle_file, variable_names, problem_name):
    """Return the oracle's ``VARIABLE_DESCRIPTIONS`` (or ``None`` if undefined)."""
    descriptions = getattr(load_module(oracle_file), "VARIABLE_DESCRIPTIONS", None)
    if descriptions is None:
        return None
    descriptions = list(descriptions)
    if len(descriptions) != len(variable_names):
        raise ValueError(
            f"VARIABLE_DESCRIPTIONS for '{problem_name}' has {len(descriptions)} "
            f"entries, but VARIABLE_NAMES has {len(variable_names)}"
        )
    if any(not isinstance(d, str) or not d.strip() for d in descriptions):
        raise ValueError(f"VARIABLE_DESCRIPTIONS for '{problem_name}' contains an empty entry")
    return tuple(descriptions)


def load_decision_variables(oracle_file, problem_name):
    """Return a dcp-bench problem's ``DECISION_VARIABLES`` prompt block."""
    value = getattr(load_module(oracle_file), "DECISION_VARIABLES", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{oracle_file}' must define a nonempty DECISION_VARIABLES string")
    return value
