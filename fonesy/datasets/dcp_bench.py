"""DCP-Bench-Open problems converted to CNF (see ``ground-truth-dimacs/dcp-bench.py``)."""

from fonesy.datasets.dummy import DimacsTargetDataset
from fonesy.datasets.prompts import DATA_DIR, load_decision_variables, render_prompt


class DcpBenchDataset(DimacsTargetDataset):
    """The prompt lists the original CP decision variables (name + domain)
    instead of one line per Boolean variable."""

    benchmark = "dcp_bench"
    data_dir = DATA_DIR / "dcp-bench"

    def load(self):
        if not (self.path / "inputs").exists():
            raise FileNotFoundError(
                f"DCP-Bench data not found in {self.path}. Run `make dcp-bench` first."
            )
        return super().load()

    def render(self, problem_text, spec, metadata_file, problem_id):
        return render_prompt(
            self.template, problem_text, load_decision_variables(metadata_file, problem_id)
        )
