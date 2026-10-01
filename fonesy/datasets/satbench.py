"""SATBench (SAT instances only), prepared by ``ground-truth-dimacs/satbench.py``."""

import json
import re
from itertools import product
from math import prod

from fonesy.datasets.base import BaseDataset, VariableSpec
from fonesy.datasets.prompts import (
    DATA_DIR,
    PROMPTS_DIR,
    format_variable_list,
    load_prompt_template,
    render_prompt,
)
from fonesy.evaluation.dimacs import verify_cnf
from fonesy.general_utils import load_file
from fonesy.statistics.statistics import CorrectionStatus

_CONDITION_ORDINAL_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
_VARIABLE_SCHEMA_RE = re.compile(r"\bx\s*\([^()\n]*\)")


def native_variable_names(dims):
    """Return SATBench source names ``x(i, j)`` and portable aliases ``x_i_j`` in row-major order."""
    source_names, variable_names = [], []
    for coordinates in product(*(range(dimension) for dimension in dims)):
        if len(coordinates) == 1:
            source_names.append(f"x({coordinates[0]},)")
        else:
            source_names.append(f"x({', '.join(map(str, coordinates))})")
        variable_names.append("x_" + "_".join(map(str, coordinates)))
    return tuple(source_names), tuple(variable_names)


def _canonicalize_variable_mapping(mapping, dims, sample_id):
    """Replace the ``x(...)`` coordinate schema in the mapping text with ``x_i_j``."""
    labels = ("i", "j", "k")
    if not 1 <= len(dims) <= len(labels):
        raise ValueError(f"SATBench sample '{sample_id}' has unsupported arity {len(dims)}")
    matches = tuple(_VARIABLE_SCHEMA_RE.finditer(mapping))
    if len(matches) != 1:
        raise ValueError(
            f"SATBench sample '{sample_id}' must contain exactly one x(...) schema; "
            f"found {len(matches)}"
        )
    match = matches[0]
    return mapping[: match.start()] + "x_" + "_".join(labels[: len(dims)]) + mapping[match.end():]


def _render_conditions(conditions):
    rendered = []
    for index, condition in enumerate(conditions, start=1):
        match = _CONDITION_ORDINAL_RE.fullmatch(condition)
        if match is not None and int(match.group(1)) == index:
            condition = match.group(2).strip()
        rendered.append(f"  {index}. {condition}")
    return "\n".join(rendered)


class SATBenchDataset(BaseDataset):
    """Each item is ``(prompt, target_dimacs, problem_text, id)``."""

    dataset_type = "dimacs"
    allow_auxiliary_variables = False

    def __init__(self, mode="zero-shot", cnf_format="dimacs", root=None):
        super().__init__()
        self.path = root if root is not None else DATA_DIR / "satbench"
        self.dataset_file = self.path / "dataset.jsonl"
        self.mode = mode
        self.cnf_format = cnf_format
        self._problem_template = load_file(PROMPTS_DIR / "satbench_problem.txt")
        self._format_template = load_prompt_template("satbench", cnf_format, mode)
        self.data = self.load()

    def _load_record(self, obj):
        sample_id = obj["id"].strip()
        dims = tuple(obj["dims"])
        if obj["num_vars"] != prod(dims):
            raise ValueError(f"SATBench sample '{sample_id}': num_vars != prod(dims)")
        source_names, variable_names = native_variable_names(dims)
        if tuple(obj["variable_names"]) != variable_names:
            raise ValueError(f"SATBench sample '{sample_id}': unexpected variable_names")
        spec = VariableSpec(source_names=source_names, variable_names=variable_names)

        target = obj["target"].strip()
        target_cnf, target_error, status = verify_cnf(target, llm_response=False)
        if target_cnf is None or status == CorrectionStatus.INVALID:
            raise ValueError(f"SATBench sample '{sample_id}' has an invalid target: {target_error}")

        problem = self._problem_template.format(
            scenario=obj["scenario"].strip(),
            variable_mapping=_canonicalize_variable_mapping(
                obj["variable_mapping"].strip(), dims, sample_id
            ),
            conditions=_render_conditions(tuple(c.strip() for c in obj["conditions"])),
        ).strip()
        prompt = render_prompt(
            self._format_template, problem, format_variable_list(variable_names)
        ).strip()
        self._variable_specs[sample_id] = spec
        return {
            "id": sample_id,
            "problem": problem,
            "input": prompt,
            "target": target,
            "variable_spec": spec,
        }

    def load(self):
        if not self.dataset_file.is_file():
            raise FileNotFoundError(
                f"SATBench dataset not found: {self.dataset_file}. Run `make satbench` first."
            )
        with self.dataset_file.open("r", encoding="utf-8") as fh:
            return [self._load_record(json.loads(line)) for line in fh if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input"], item["target"], item["problem"], item["id"]
