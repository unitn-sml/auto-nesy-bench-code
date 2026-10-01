"""Name -> class factories for LLM backends, SPL backbones and SPL tasks."""

from fonesy.llms.dummy import DummyLLM
from fonesy.llms.vllm_llm import VllmLLM


class LLMFactory:
    @staticmethod
    def get_llm(name: str, model_path: str = None, **kwargs):
        name = name.lower()
        if name == "dummy":
            return DummyLLM(cnf_format=kwargs.get("cnf_format", "dimacs"))
        if name == "vllm":
            return VllmLLM(model_path=model_path or "Qwen/Qwen3-8B", **kwargs)
        raise ValueError(f"Unknown LLM: {name}")


class NetworkFactory:
    @staticmethod
    def get_network(name: str, n_classes: int, in_channels: int, **kwargs):
        """Instantiate a backbone; extra *kwargs* go to its constructor."""
        from fonesy.networks.bert import BertBackbone
        from fonesy.networks.conv4 import Conv4
        from fonesy.networks.fasterrcnn import FasterRCNNBackbone
        from fonesy.networks.mlp import MLPBackbone
        from fonesy.networks.resnet_scratch import ResNet18Scratch
        from fonesy.networks.slowfast import SlowFastBackbone

        networks = {
            "conv4": Conv4,
            "resnet18": ResNet18Scratch,
            "mlp": MLPBackbone,
            "fasterrcnn": FasterRCNNBackbone,
            "slowfast": SlowFastBackbone,
            "bert": BertBackbone,
        }
        if name.lower() not in networks:
            raise ValueError(f"Unknown network: {name}")
        return networks[name.lower()](n_classes=n_classes, in_channels=in_channels, **kwargs)


class TaskFactory:
    @staticmethod
    def get_task(name: str, root: str, **kwargs):
        """Instantiate the data of an SPL task; *root* is the data directory."""
        from fonesy.datasets.tasks.bdd_oia import BddOiaTask
        from fonesy.datasets.tasks.bdd_oia_simple import BddOiaSimpleTask
        from fonesy.datasets.tasks.cebab import CeBaBTask
        from fonesy.datasets.tasks.chx import ChxTask
        from fonesy.datasets.tasks.cifar100_mut_ex import Cifar100MutExTask
        from fonesy.datasets.tasks.cifar10_concept import Cifar10ConceptTask
        from fonesy.datasets.tasks.cle4evr import CLE4evrTask
        from fonesy.datasets.tasks.fashion_mut_ex import FashionMutExTask
        from fonesy.datasets.tasks.kand_logic import KandLogicSimpleTask, KandLogicTask
        from fonesy.datasets.tasks.mnist_addition import MnistAdditionTask
        from fonesy.datasets.tasks.mnist_multiplication import MnistMultiplicationTask
        from fonesy.datasets.tasks.road_r import RoadRTask
        from fonesy.datasets.tasks.sudoku import SudokuTask
        from fonesy.datasets.tasks.sushi import SushiTask
        from fonesy.datasets.tasks.warcraft_path import WarcraftPathTask

        tasks = {
            "mnist-addition": MnistAdditionTask,
            "mnist-multiplication": MnistMultiplicationTask,
            "bdd-oia": BddOiaTask,
            "bdd-oia-simple": BddOiaSimpleTask,
            "sudoku": SudokuTask,
            "cle4evr": CLE4evrTask,
            "road-r": RoadRTask,
            "kand-logic": KandLogicTask,
            "kand-logic-simple": KandLogicSimpleTask,
            "sushi": SushiTask,
            "fashion-mut-ex": FashionMutExTask,
            "cifar10": Cifar10ConceptTask,
            "cifar100-mut-ex": Cifar100MutExTask,
            "warcraft-path": WarcraftPathTask,
            "cebab": CeBaBTask,
            "chx": ChxTask,
        }
        if name.lower() not in tasks:
            raise ValueError(f"Unknown task: {name}")
        return tasks[name.lower()](root=root)
