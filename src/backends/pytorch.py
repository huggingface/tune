from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger

import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer, TensorType

from backends import Backend, BackendConfig
from benchmark import Benchmark
from config import BenchmarkConfig

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    use_torchscript: bool = False


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        self.model = AutoModel.from_pretrained(model)

        LOGGER.info(f"Allocated PyTorch Backend for model: {model}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        backend = cls(config.model)
        backend.configure(config.backend)

        return backend

    def configure(self, config: PyTorchConfig):
        LOGGER.info("Configuring PyTorch Benchmark:")

        # Disable gradients
        torch.set_grad_enabled(False)
        LOGGER.info("\t+ Disabled gradients")

        torch.set_num_threads(config.num_threads)
        LOGGER.info(f"\t+ Number of threads (torch.set_num_threads({config.num_threads}))")

        # TODO: Setting this value multiple times between PyTorch & TorchScript runs raise a C error
        # torch.set_num_interop_threads(config.num_interops_threads)
        LOGGER.info(
            f"\t+ Number of interop threads (torch.set_num_interop_threads({config.num_interops_threads}))"
        )

        self.model.eval()
        LOGGER.info("\t+ Turning eval mode on Module (model.eval())")

        if config.use_torchscript:
            self.model.config.return_dict = False
            LOGGER.info("\t+ Disabling dictionary output for TorchScript")

    def execute(self, config: BenchmarkConfig) -> Benchmark:
        if config.backend.use_torchscript:
            return self._run_torchscript(config)
        else:
            return self._run_pytorch(config)

    def _run_pytorch(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        LOGGER.info("Running PyTorch Eager benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.PYTORCH,
        )

        inputs = inputs.to(config.device)
        self.model = self.model.to(config.device)

        # Warmup
        for _ in trange(config.warmup_runs, desc="Warming up"):
            self.model(**inputs)

        # Run benchmark
        for _ in trange(config.num_runs, desc="Running benchmark"):
            with benchmark.track():
                self.model(**inputs)

        return benchmark

    def _run_torchscript(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        LOGGER.info("Running TorchScript benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.PYTORCH,
        )

        inputs.to(config.device)
        self.model = self.model.to(config.device)

        # To be sure inputs will be presented with the right prototype
        ordered_inputs = OrderedDict({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "token_type_ids": inputs.token_type_ids,
        })

        LOGGER.debug("Calling torch JIT on model (optimize=True)")
        model_scripted = torch.jit.trace(self.model, tuple(ordered_inputs.values()))

        with torch.jit.optimized_execution(True):
            for _ in trange(config.warmup_runs, desc="Warming up"):
                model_scripted(*ordered_inputs.values())

            for _ in trange(config.num_runs, desc="Running benchmark"):
                with benchmark.track():
                    model_scripted(*ordered_inputs.values())

        return benchmark

