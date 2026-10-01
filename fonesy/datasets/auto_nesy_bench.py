"""auto-nesy-bench: NeSy tasks whose ground truth is a Python oracle + DIMACS."""

from fonesy.datasets.base import BaseDataset
from fonesy.datasets.prompts import (
    DATA_DIR,
    format_variable_list,
    load_prompt_template,
    load_variable_descriptions,
    load_variable_spec,
)
from fonesy.general_utils import load_file


class AutoNesyBenchDataset(BaseDataset):
    """Each item is ``(prompt, oracle_path, problem_text, id)``.

    ``detailed`` selects the detailed (``inputs_detailed/``) or the
    non-detailed (``inputs_non_detailed/``) constraint descriptions.
    """

    dataset_type = "program"

    def __init__(self, mode="zero-shot", cnf_format="dimacs", detailed=True, root=None):
        super().__init__()
        self.path = root if root is not None else DATA_DIR / "auto-nesy-bench"
        self.targets_dir = self.path / "targets"
        self.mode = mode
        self.cnf_format = cnf_format
        self.detailed = detailed
        self.template = load_prompt_template("auto_nesy_bench", cnf_format, mode)
        self.data = self.load()

    def load(self):
        inputs_dir = self.path / ("inputs_detailed" if self.detailed else "inputs_non_detailed")
        data = []
        for problem_file in sorted(inputs_dir.glob("*.txt")):
            problem_id = problem_file.stem
            problem_text = load_file(problem_file)
            oracle_file = self.targets_dir / f"{problem_id}.py"
            target_text = load_file(self.targets_dir / f"{problem_id}.dimacs")
            spec = load_variable_spec(oracle_file, problem_id, target_text)
            self._variable_specs[problem_id] = spec
            descriptions = load_variable_descriptions(oracle_file, spec.variable_names, problem_id)
            prompt = self.template.replace(
                "{variables}", format_variable_list(spec.variable_names, descriptions)
            ).replace("{problem}", problem_text)
            data.append(
                {"input": prompt, "id": problem_id, "problem": problem_text, "variable_spec": spec}
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x["input"], str(self.targets_dir / f"{x['id']}.py"), x["problem"], x["id"]
