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
from pathlib import Path

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode
from tqdm import trange
from transformers import TensorType
from transformers.convert_graph_to_onnx import convert as onnx_convert

from backends import BackendConfig, Backend
from benchmark import Benchmark


ALL_GRAPH_OPTIMIZATION_LEVELS = {
    GraphOptimizationLevel.ORT_ENABLE_ALL,
    GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    GraphOptimizationLevel.ORT_ENABLE_BASIC,
    GraphOptimizationLevel.ORT_DISABLE_ALL
}

ALL_EXECUTION_MODE = {
    ExecutionMode.ORT_PARALLEL,
    ExecutionMode.ORT_SEQUENTIAL
}


@dataclass
class OnnxRuntimeConfig(BackendConfig):
    graph_optimisation_level: str = "ORT_ENABLE_ALL"
    execution_mode: str = "ORT_PARALLEL"
    name: str = "onnxruntime"
    opset: int = 12


class OnnxRuntimeBackend(Backend[OnnxRuntimeConfig]):

    def __init__(self, model: str, onnx_path: str):
        super().__init__(model)

        self.onnx_path = onnx_path
        self.session_opts = SessionOptions()

    @staticmethod
    def convert(model: str, output: Path, opset: int = 12) -> Path:
        if output.exists():
            return output

        onnx_convert("pt", model, output, opset=opset)

    @classmethod
    def allocate(cls, config: 'BenchmarkConfig'):
        onnx_model_path = Path(f"onnx_graphs/{config.model}.onnx")
        OnnxRuntimeBackend.convert(config.model, onnx_model_path, config.backend.opset)

        return OnnxRuntimeBackend(config.model, onnx_model_path.absolute().as_posix())

    def configure(self, config: OnnxRuntimeConfig):
        assert config.graph_optimisation_level in ALL_GRAPH_OPTIMIZATION_LEVELS, f"Unknown {config.graph_optimisation_level}"
        assert config.execution_mode in ALL_EXECUTION_MODE, f"Unknown {config.execution_mode}"

        self.session_opts.execution_mode = config.execution_mode
        self.session_opts.graph_optimization_level = config.graph_optimisation_level
        self.session_opts.inter_op_num_threads = config.num_interops_threads
        self.session_opts.intra_op_num_threads = config.num_threads

    def execute(self, config: 'BenchmarkConfig') -> Benchmark:
        benchmark = Benchmark()
        session = InferenceSession(self.onnx_path, self.session_opts)

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
        for _ in trange(config.num_runs, desc="Running benchmark"):
            with benchmark.track():
                session.run(None, inputs)
        return benchmark

    def clean(self, config: 'BenchmarkConfig'):
        onnx_path = Path(self.onnx_path)

        if onnx_path.exists():
            # Care for external data format (multiple file) if exporting bigger model
            onnx_path.unlink()
