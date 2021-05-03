#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import contextlib
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import Optional, Tuple, Callable, List, Set

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras import Input
from tqdm import trange
from transformers import PreTrainedTokenizer, TFAutoModel, TFBertModel, TFPreTrainedModel, TensorType

from backends import Backend, BackendConfig
from benchmark import Benchmark
from config import BenchmarkConfig
from utils import SEC_TO_NS_SCALE


BACKEND_NAME = "tensorflow"
LOGGER = getLogger("tensorflow")


def get_tf_device(device: str) -> str:
    if device == "cuda":
        if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            raise ValueError(f"No GPU detected, cannot move data to {device}")
        return tf.DeviceSpec(device_type="GPU")
    else:
        return tf.DeviceSpec(device_type="CPU")


def as_saved_model(tokenizer: PreTrainedTokenizer, model: TFPreTrainedModel, inputs: List, saved_model_path: Path, flag: str = "tune") -> Path:
    encodings = tokenizer(inputs, is_split_into_words=True, return_tensors="tf")

    # Generate symbolic trace
    tf_inputs = {name: Input((None, ), batch_size=None, dtype=tf.int32, name=name) for name, value in encodings.items()}
    tf_outputs = model(tf_inputs)
    tf_model = tf.keras.models.Model(inputs=tf_inputs, outputs={"output": tf_outputs[0]})

    # Saved SavedModel
    tf.saved_model.save(tf_model, saved_model_path.as_posix())

    # Generate a flag file indicating this folder was generated from the tune framework
    saved_model_path.joinpath(flag).touch()
    return saved_model_path


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


@dataclass
class TensorflowConfig(BackendConfig):
    name: str = "tensorflow"
    use_xla: bool = False
    use_saved_model_format: bool = False
    eager_mode: bool = True
    experimental_compiler: Optional[bool] = None
    local_model_path: Optional[str] = None

    @staticmethod
    def version() -> str:
        return tf.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union({
            "use_xla",
            "eager_mode",
            "experimental_compiler",
            "use_saved_model_format",
            "local_model_path"
        })


class TensorflowBackend(Backend[TensorflowConfig]):
    NAME = BACKEND_NAME
    SAVED_MODEL_PATH = "saved_model"

    def __init__(self, model: str, local_model_path: str = None):
        super().__init__(model)
        self.model = model
        self.model_info = None  # Only used when working with SavedModel
        self.local_model_path = local_model_path  # Local

        LOGGER.info(f"Allocated TensorFlow Backend for model: {model}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        if config.backend.use_saved_model_format and "@" in config.model:
            model_name, model_path = config.model.split("@")
            LOGGER.info(f"Local SavedModel format detected: model={model_name}, path={model_path}")

            backend = TensorflowBackend(model_name, model_path)
        else:
            backend = TensorflowBackend(config.model)

        backend.configure(config.backend)

        return backend

    def clean(self, config: 'BenchmarkConfig'):
        saved_model_path = Path(TensorflowBackend.SAVED_MODEL_PATH)
        if saved_model_path.exists() and saved_model_path.joinpath("tune"):
            LOGGER.debug(f"Cleaning SavedModel folder at {saved_model_path}")
            rmtree(saved_model_path)
            # saved_model_path.rmdir()

    def configure(self, config: TensorflowConfig):
        super().configure(config)

        LOGGER.info("Configuring TensorFlow Benchmark:")

        # Eager execution should only be tuned for TensorFlow not for XLA
        if config.name == "tensorflow" and not config.eager_mode:
            LOGGER.info(
                "\t+ Disabling eager execution"
            )
            tf.compat.v1.disable_eager_execution()

        if config.num_threads is not None:
            if tf.config.threading.get_intra_op_parallelism_threads() != config.num_threads:
                tf.config.threading.set_intra_op_parallelism_threads(config.num_threads)

            LOGGER.info(
                f"\t+ Number of intra op threads ("
                f"tf.config.threading.set_intra_op_parallelism_threads("
                f"{tf.config.threading.get_intra_op_parallelism_threads()}"
                f"))"
            )

        if config.num_interops_threads is not None:
            if tf.config.threading.get_inter_op_parallelism_threads() != config.num_interops_threads:
                tf.config.threading.set_inter_op_parallelism_threads(config.num_interops_threads)

            LOGGER.info(
                f"\t+ Number of inter op threads ("
                f"tf.config.threading.set_inter_op_parallelism_threads("
                f"{tf.config.threading.get_inter_op_parallelism_threads()}"
                f"))"
            )

        # If we need to use the model as SavedModel format
        if config.use_saved_model_format:

            # Local model support
            if self.local_model_path is None:
                LOGGER.info(f"Converting model: {self.model} to SavedModel format")
                with options({
                    "constant_folding": True,
                    "shape_optimization": True,
                    "disable_model_pruning": False,
                    "arithmetic_optimization": True,
                    "function_optimization": True
                }):
                    with tf.device("CPU"):
                        model = TFAutoModel.from_pretrained(self.model)
                        self.local_model_path = as_saved_model(
                            tokenizer=self.tokenizer,
                            model=model,
                            inputs=self._get_dummy_inputs(
                                1, model.config.max_position_embeddings - self.tokenizer.num_special_tokens_to_add()
                            ),
                            saved_model_path=Path(TensorflowBackend.SAVED_MODEL_PATH),
                            flag="tune"
                        )

                    LOGGER.debug(f"Converted SavedModel stored at {self.local_model_path}")

            # Load the model
            saved_model_path = Path(self.local_model_path)
            LOGGER.info(f"Loading SavedModel from {saved_model_path}")
            self.model_info = tf.saved_model.load(saved_model_path.as_posix())
            self.model = self.model_info.signatures["serving_default"]
        else:
            # Postponing model allocation to tune intra/inter ops before executing any other TF related code.
            self.model = TFAutoModel.from_pretrained(self.model)

    def execute(self, config: BenchmarkConfig, is_reference: bool = False) -> Tuple[Benchmark, np.ndarray]:
        if not config.backend.use_xla:
            return self._run_tf(config, is_reference)
        else:
            return self._run_xla(config, is_reference)

    def _run_tf(self, config: BenchmarkConfig, is_reference: bool) -> Tuple[Benchmark, np.ndarray]:
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

            # Move tf.constants to GPU ... https://github.com/tensorflow/tensorflow/issues/42242#issuecomment-675590057
            inputs = {name: tf.identity(t) for name, t in inputs.items()}

            # SavedModel concrete function needs unwrapped arguments ...
            model_f = lambda x: self.model(**x) if config.backend.use_saved_model_format else self.model

            # Warmup
            outputs = []
            for _ in trange(config.warmup_runs, desc="Warming up"):
                output = model_f(inputs)
                outputs.append(output.last_hidden_state.numpy())

            # Let's not run the benchmark for the reference backend,
            # as we are more interested in the output tensors.
            if not is_reference:

                # Run benchmark
                benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
                while sum(benchmark.latencies) < benchmark_duration_ns:
                    with benchmark.track():
                        model_f(inputs)

                benchmark.finalize(benchmark_duration_ns)

            return benchmark, np.stack(outputs)

    def _run_xla(self, config: BenchmarkConfig, is_reference: bool) -> Tuple[Benchmark, np.ndarray]:
        @tf.function(experimental_compile=config.backend.experimental_compiler)
        def xla_model(inputs):
            # SavedModel concrete function needs unwrapped arguments ...
            if config.backend.use_saved_model_format:
                return self.model(**inputs)
            else:
                return self.model(inputs)

        LOGGER.info("Running TensorFlow XLA benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        with tf.device(get_tf_device(config.device)):
            with options({
                "constant_folding": True,
                "shape_optimization": True,
                "disable_model_pruning": False,
                "arithmetic_optimization": True,
                "function_optimization": True
            }):
                inputs = self.tokenizer(
                    dummy_inputs,
                    is_split_into_words=True,
                    return_tensors=TensorType.TENSORFLOW,
                )

                # Move tf.constants to GPU ...
                # https://github.com/tensorflow/tensorflow/issues/42242#issuecomment-675590057
                inputs = {name: tf.identity(t) for name, t in inputs.items()}

                # Warmup
                outputs = []
                for _ in trange(config.warmup_runs, desc="Warming up"):
                    output = xla_model(inputs)
                    outputs.append(output.last_hidden_state.numpy())

                # Let's not run the benchmark for the reference backend,
                # as we are more interested in the output tensors.
                if not is_reference:

                    # Run benchmark
                    benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
                    while sum(benchmark.latencies) < benchmark_duration_ns:
                        with benchmark.track():
                            xla_model(inputs)

                    benchmark.finalize(benchmark_duration_ns)
        return benchmark, np.stack(outputs)
