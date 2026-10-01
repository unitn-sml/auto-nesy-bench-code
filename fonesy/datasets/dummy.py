"""Toy propositional problems used to smoke-test the pipeline."""

from fonesy.datasets.base import BaseDataset
from fonesy.datasets.prompts import (
    DATA_DIR,
    format_variable_list,
    load_prompt_template,
    load_variable_spec,
    render_prompt,
)
from fonesy.general_utils import load_file


class DimacsTargetDataset(BaseDataset):
    """Problems stored as ``inputs/<id>.txt`` + ``targets/<id>.{dimacs,py}``.

    Each item is ``(prompt, target_dimacs, problem_text, id)``.
    """

    dataset_type = "dimacs"
    benchmark = None
    data_dir = None

    def __init__(self, mode="zero-shot", cnf_format="dimacs", root=None):
        super().__init__()
        self.path = root if root is not None else self.data_dir
        self.mode = mode
        self.cnf_format = cnf_format
        self.template = load_prompt_template(self.benchmark, cnf_format, mode)
        self.data = self.load()

    def render(self, problem_text, spec, metadata_file, problem_id):
        names = spec.variable_names[: spec.primary_num_vars]
        return render_prompt(self.template, problem_text, format_variable_list(names))

    def load(self):
        inputs_dir, targets_dir = self.path / "inputs", self.path / "targets"
        if not inputs_dir.exists():
            raise FileNotFoundError(f"Missing problems directory: {inputs_dir}")
        data = []
        for problem_file in sorted(inputs_dir.glob("*.txt")):
            problem_id = problem_file.stem
            problem_text = load_file(problem_file)
            target_text = load_file(targets_dir / f"{problem_id}.dimacs")
            metadata_file = targets_dir / f"{problem_id}.py"
            spec = load_variable_spec(metadata_file, problem_id, target_text)
            self._variable_specs[problem_id] = spec
            data.append(
                {
                    "input": self.render(problem_text, spec, metadata_file, problem_id).strip(),
                    "target": target_text,
                    "problem": problem_text,
                    "id": problem_id,
                    "variable_spec": spec,
                }
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x["input"], x["target"], x["problem"], x["id"]


class DummyDataset(DimacsTargetDataset):
    benchmark = "dummy"
    data_dir = DATA_DIR / "dummy"
