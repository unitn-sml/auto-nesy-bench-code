"""
Dummy LLM for testing.
"""

from fonesy.llms.base import BaseLLM


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models."""
    pass


class DummyLLM(BaseLLM):
    def __init__(self, name: str = "dummy", cnf_format: str = "dimacs", **kwargs):
        super().__init__(name)
        self.cnf_format = cnf_format

    def generate(self, prompt: str, **kwargs) -> str:
        if self.cnf_format == "nat":
            return "<cnf>\n(x1 OR x2 OR NOT x3) AND (NOT x2 OR x3)\n</cnf>\n"
        if self.cnf_format == "pysat":
            return (
                "<cnf>\n"
                "from pysat.formula import CNF\n"
                "import sys\n"
                "\n"
                "cnf = CNF()\n"
                "cnf.append([1, 2, -3])\n"
                "cnf.append([-2, 3])\n"
                "cnf.to_fp(sys.stdout)\n"
                "</cnf>\n"
            )
        # Deterministic mock output (DIMACS)
        return """<cnf>
c Example CNF
p cnf 3 2
1 2 -3 0
-2 3 0
</cnf>
"""


def configure_subparsers(subparsers):
    """Configure subparsers."""
    dummy = subparsers.add_parser(
        "dummy",
        help="Dummy LLM",
    )

    configure_global_arguments(dummy)
