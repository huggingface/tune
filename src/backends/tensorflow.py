from dataclasses import dataclass
from logging import getLogger

import tensorflow as tf
from tqdm import trange
from transformers import TFAutoModel, TensorType

from backends import Backend, BackendConfig, BackendConfigT
from benchmark import Benchmark
from config import BenchmarkConfig

BACKEND_NAME = "tensorflow"
LOGGER = getLogger("tensorflow")


def get_tf_device(device: str) -> str:
    if device == "cuda":
        if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            raise ValueError(f"No GPU detected, cannot move data to {device}")
        return "/GPU:0"
    else:
        return "/CPU"


@dataclass
class TensorflowConfig(BackendConfig):
    name: str = "tensorflow"
    use_xla: bool = False


class TensorflowBackend(Backend[TensorflowConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        self.model = TFAutoModel.from_pretrained(model)

        LOGGER.info(f"Allocated TensorFlow Backend for model: {model}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        return TensorflowBackend(config.model)

    def configure(self, config: BackendConfigT):
        LOGGER.info("Configuring TensorFlow Benchmark:")

        tf.config.threading.set_inter_op_parallelism_threads(config.num_interops_threads)
        LOGGER.info(
            f"\t+ Number of inter op threads ("
            f"tf.config.threading.set_inter_op_parallelism_threads({config.num_interops_threads})"
            f")"
        )

        tf.config.threading.set_intra_op_parallelism_threads(config.num_threads)
        LOGGER.info(
            f"\t+ Number of intra op threads ("
            f"tf.config.threading.set_intra_op_parallelism_threads({config.num_threads})"
            f")"
        )

    def execute(self, config: BenchmarkConfig) -> Benchmark:
        if not config.backend.use_xla:
            return self._run_eager(config)
        else:
            return self._run_xla(config)

    def _run_eager(self, config: BenchmarkConfig) -> Benchmark:
        LOGGER.info("Running TensorFlow Eager benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        with tf.device(get_tf_device(config.device)):
            inputs = self.tokenizer(
                dummy_inputs,
                is_split_into_words=True,
                return_tensors=TensorType.TENSORFLOW,
            )

            # Warmup
            for _ in trange(config.warmup_runs, desc="Warming up"):
                self.model(inputs)

            # Run benchmark
            for _ in trange(config.num_runs, desc="Running benchmark"):
                with benchmark.track():
                    self.model(inputs)
            return benchmark

    def _run_xla(self, config: BenchmarkConfig) -> Benchmark:
        @tf.function()
        def xla_model(inputs):
            return self.model(inputs)

        LOGGER.info("Running TensorFlow XLA benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        with tf.device(get_tf_device(config.device)):
            inputs = self.tokenizer(
                dummy_inputs,
                is_split_into_words=True,
                return_tensors=TensorType.TENSORFLOW,
            )

            # Warmup
            for _ in trange(config.warmup_runs, desc="Warming up"):
                xla_model(inputs)

            # Run benchmark
            for _ in trange(config.num_runs, desc="Running benchmark"):
                with benchmark.track():
                    xla_model(inputs)

        return benchmark