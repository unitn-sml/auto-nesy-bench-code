"""Benchmarks evaluated by the ``evaluate`` sub-command."""

from fonesy.datasets.auto_nesy_bench import AutoNesyBenchDataset
from fonesy.datasets.base import BaseDataset, VariableSpec
from fonesy.datasets.dcp_bench import DcpBenchDataset
from fonesy.datasets.dummy import DummyDataset
from fonesy.datasets.prompts import MODE_CHOICES
from fonesy.datasets.satbench import SATBenchDataset
from fonesy.llms import dummy, vllm_llm

DATASETS = {
    "dummy": (DummyDataset, "Toy propositional problems (pipeline smoke test)"),
    "auto-nesy-bench": (AutoNesyBenchDataset, "auto-nesy-bench NeSy tasks"),
    "dcp-bench": (DcpBenchDataset, "DCP-Bench-Open problems converted to CNF"),
    "satbench": (SATBenchDataset, "SATBench satisfiable instances"),
}


def configure_subparsers(subparsers):
    """Register one sub-command per dataset, each followed by an LLM backend."""
    for name, (_, help_text) in DATASETS.items():
        parser = subparsers.add_parser(name, help=help_text)
        parser.add_argument(
            "--mode",
            choices=MODE_CHOICES,
            default="zero-shot",
            help="Prompting strategy: zero-shot (ZS) or zero-shot chain-of-thought (ZS-CoT).",
        )
        if name == "auto-nesy-bench":
            parser.add_argument(
                "--non-detailed",
                action="store_true",
                default=False,
                help="Use the non-detailed constraint descriptions instead of the detailed ones.",
            )
        llm_subparsers = parser.add_subparsers(dest="llm")
        dummy.configure_subparsers(llm_subparsers)
        vllm_llm.configure_subparsers(llm_subparsers)


def get_dataset(args):
    """Instantiate the dataset selected on the command line."""
    cls = DATASETS[args.dataset][0]
    kwargs = {"mode": args.mode, "cnf_format": getattr(args, "cnf_format", "dimacs")}
    if args.dataset == "auto-nesy-bench":
        kwargs["detailed"] = not args.non_detailed
    return cls(**kwargs)


__all__ = [
    "AutoNesyBenchDataset",
    "BaseDataset",
    "DATASETS",
    "DcpBenchDataset",
    "DummyDataset",
    "SATBenchDataset",
    "VariableSpec",
    "configure_subparsers",
    "get_dataset",
]
