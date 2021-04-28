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

from dataclasses import dataclass
from logging import getLogger
from os import getpid
from pathlib import Path

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode, __version__ as ort_version
from onnxruntime_tools.transformers.optimizer import optimize_model
from tqdm import trange
from transformers import TensorType
from transformers.convert_graph_to_onnx import convert as onnx_convert

from backends import BackendConfig, Backend
from benchmark import Benchmark
from utils import SEC_TO_NS_SCALE


ALL_GRAPH_OPTIMIZATION_LEVELS = {
    GraphOptimizationLevel.ORT_ENABLE_ALL,
    GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    GraphOptimizationLevel.ORT_ENABLE_BASIC,
    GraphOptimizationLevel.ORT_DISABLE_ALL
}
ALL_GRAPH_OPTIMIZATION_LEVELS_FROM_STR = {
    level.name: level
    for level in ALL_GRAPH_OPTIMIZATION_LEVELS
}

ALL_EXECUTION_MODE = {
    ExecutionMode.ORT_PARALLEL,
    ExecutionMode.ORT_SEQUENTIAL
}

ALL_EXECUTION_MODE_FROM_STR = {
    level.name: level
    for level in ALL_EXECUTION_MODE
}


@dataclass
class OnnxRuntimeConfig(BackendConfig):
    name: str = "onnxruntime"
    opset: int = 12
    graph_optimisation_level: str = "ORT_ENABLE_ALL"
    execution_mode: str = "ORT_PARALLEL"

    @staticmethod
    def version() -> str:
        return ort_version


BACKEND_NAME = "onnxruntime"
LOGGER = getLogger(BACKEND_NAME)
ONNX_GRAPHS_FOLDER = "onnx_graphs"


class OnnxRuntimeBackend(Backend[OnnxRuntimeConfig]):

    def __init__(self, model: str, onnx_path: str):
        super().__init__(model)

        self.onnx_path = onnx_path
        self.optimized_onnx_graph = None
        self.session_opts = SessionOptions()

    @staticmethod
    def convert(model: str, output: Path, opset: int = 12) -> Path:
        if output.exists():
            return output

        onnx_convert("pt", model, output, opset=opset)

    @classmethod
    def allocate(cls, config: 'BenchmarkConfig'):
        onnx_model_path = Path(f"{ONNX_GRAPHS_FOLDER}/{config.model}.onnx.{getpid()}")
        OnnxRuntimeBackend.convert(config.model, onnx_model_path, config.backend.opset)

        backend = OnnxRuntimeBackend(config.model, onnx_model_path.absolute().as_posix())
        backend.configure(config.backend)
        return backend

    def configure(self, config: OnnxRuntimeConfig):
        assert config.graph_optimisation_level in ALL_GRAPH_OPTIMIZATION_LEVELS_FROM_STR, f"Unknown {config.graph_optimisation_level}"
        assert config.execution_mode in ALL_EXECUTION_MODE_FROM_STR, f"Unknown {config.execution_mode}"

        super().configure(config)

        LOGGER.info("Configuring ONNX Runtime Benchmark:")

        self.session_opts.execution_mode = ALL_EXECUTION_MODE_FROM_STR[config.execution_mode]
        LOGGER.info(f"\t- Setting Execution Mode: {self.session_opts.execution_mode}")

        self.session_opts.graph_optimization_level = ALL_GRAPH_OPTIMIZATION_LEVELS_FROM_STR[config.graph_optimisation_level]
        LOGGER.info(f"\t- Setting Graph Optimization Level: {self.session_opts.graph_optimization_level}")

        if config.num_threads is not None:
            if self.session_opts.intra_op_num_threads != config.num_threads:
                self.session_opts.intra_op_num_threads = config.num_threads

            LOGGER.info(f"\t- Setting intra_op_num_threads({self.session_opts.intra_op_num_threads})")

        if config.num_interops_threads is not None:
            if self.session_opts.inter_op_num_threads != config.num_interops_threads:
                self.session_opts.inter_op_num_threads = config.num_interops_threads

            LOGGER.info(f"\t- Setting inter_op_num_threads({self.session_opts.inter_op_num_threads})")

    def execute(self, config: 'BenchmarkConfig') -> Benchmark:
        benchmark = Benchmark()

        try:
            model_opt_path = Path(self.onnx_path)
            opt_onnx_path = model_opt_path.with_suffix(".opt" + model_opt_path.suffix)

            model_opt = optimize_model(
                self.onnx_path,
                model_type="bert",
                opt_level=int(self.session_opts.graph_optimization_level)
            )
            model_opt.save_model_to_file(opt_onnx_path.absolute().as_posix())
            self.optimized_onnx_graph = opt_onnx_path.absolute().as_posix()
        except Exception as e:
            LOGGER.error(f"Unable to optimize ONNX BERT model: {e}")

        session = InferenceSession(self.optimized_onnx_graph or self.onnx_path, self.session_opts)

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.NUMPY,
        )
        inputs = {k: v.astype("i8") for k, v in inputs.items()}

        # Warmup
        for _ in trange(config.warmup_runs, desc="Warming up"):
            session.run(None, inputs)

        # Run benchmark
        benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
        while sum(benchmark.latencies) < benchmark_duration_ns:
            with benchmark.track():
                session.run(None, inputs)

        benchmark.finalize(benchmark_duration_ns)
        return benchmark

    def clean(self, config: 'BenchmarkConfig'):
        onnx_path = Path(ONNX_GRAPHS_FOLDER)

        if onnx_path.exists():
            for file in onnx_path.iterdir():
                LOGGER.debug(f"Cleaning ONNX model: {file}")
                file.unlink()

        # if Path(onnx_path).exists():
        #     # Care for external data format (multiple file) if exporting bigger model
        #     LOGGER.debug(f"Cleaning ONNX model: {self.onnx_path}")
        #     onnx_path.unlink()
        #
        # if self.optimized_onnx_graph is not None and Path(self.optimized_onnx_graph).exists():
        #     LOGGER.debug(f"Cleaning optimized ONNX model: {self.optimized_onnx_graph}")
        #     Path(self.optimized_onnx_graph).unlink()
