from dataclasses import dataclass
from logging import getLogger
from typing import List

import torch
from omegaconf import MISSING
from tqdm import trange
from transformers import AutoModel, AutoTokenizer, BatchEncoding, TensorType

from backends import Backend, BackendConfig
from benchmark import Benchmark
from config import BenchmarkConfig

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    use_torchscript: bool = MISSING


class PyTorchBackend(Backend):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

        LOGGER.info(f"Allocated PyTorch Backend for model: {model}")

    def configure(self, config: BenchmarkConfig):
        LOGGER.info("Configuring PyTorch Benchmark:")

        # Disable gradients
        torch.set_grad_enabled(False)
        LOGGER.info("\t+ Disabled gradients")

        torch.set_num_threads(config.backend.num_threads)
        LOGGER.info(f"\t+ Number of threads (torch.set_num_threads({config.backend.num_threads}))")

        torch.set_num_interop_threads(config.backend.num_interops_threads)
        LOGGER.info(
            f"\t+ Number of interop threads (torch.set_num_interop_threads({config.backend.num_interops_threads}))"
        )

        self.model.eval()
        LOGGER.info("\t+ Turning eval mode on Module (model.eval())")

    def execute(self, config: BenchmarkConfig) -> Benchmark:
        if not isinstance(config.backend, PyTorchConfig):
            raise ValueError(f"config.backend should be instance of PytorchConfig (got: {type(config.backend)})")

        self.configure(config)

        if config.backend.use_torchscript:
            return self._run_torchscript(config)
        else:
            return self._run_pytorch(config)

    def _run_pytorch(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        benchmark = Benchmark()
        inputs = self.tokenizer.prepare_for_model(
            [self.tokenizer.pad_token_id * config.sequence_length],
            return_tensors=TensorType.PYTORCH
        )

        # Warmup
        for _ in trange(config.warmup_runs, description="Warming up"):
            self.model(**inputs)

        # Run benchmark
        for _ in trange(config.num_runs, description="Running benchmark"):
            with benchmark.track():
                self.model(**inputs)
        return benchmark

    def _run_torchscript(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        raise NotImplementedError("TorchScript support is not yet implemented")

