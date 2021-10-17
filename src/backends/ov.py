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
import os

from dataclasses import dataclass
from logging import getLogger
from os import getpid
from pathlib import Path
from typing import Set, Optional, Tuple
import subprocess

import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode, __version__ as ort_version
from onnxruntime.transformers.optimizer import optimize_model
from tqdm import trange
from transformers import TensorType
from transformers.convert_graph_to_onnx import convert as onnx_convert

from backends import BackendConfig, Backend
from benchmark import Benchmark
from utils import SEC_TO_NS_SCALE

from openvino.inference_engine import IECore, IENetwork

@dataclass
class OpenVINORuntimeConfig(BackendConfig):
    name: str = "openvino"
    opset: int = 12
    api: str = "SYNC"
    pin: str = "YES"

    @staticmethod
    def version() -> str:
        return ie.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union({"api", "pin"})


BACKEND_NAME = "openvino"
LOGGER = getLogger(BACKEND_NAME)
ONNX_GRAPHS_FOLDER = "onnx_graphs"


class OpenVINORuntimeBackend(Backend[OpenVINORuntimeConfig]):

    def __init__(self, model: str, onnx_path: str, ov_path: str):
        super().__init__(model)

        self.onnx_path = onnx_path
        self.ov_path = ov_path
        self.optimized_onnx_graph = None
        self.config = {}

    @staticmethod
    def convert(config: 'BenchmarkConfig', model: str, output: Path, output_ov: Path, opset: int = 12) -> Path:
        if output.exists():
            return output

        model_path = Path(f"{ONNX_GRAPHS_FOLDER}/{model}.{getpid()}.ov/{model}")
        # Get only the model name and not the company affiliation
        model_xml = str(output_ov) + '/' + model + '.xml'
        print('Model path:{}'.format(model_path))

        if output_ov.exists():
            return model_path

        onnx_output = str(output).split('.onnx')[0]
        onnx_cmd = f"python -m transformers.onnx \
            --model {config.model} \
            {Path(onnx_output)}"
        print(onnx_cmd)

        results = subprocess.check_output(onnx_cmd, shell=True)
        
        # Setup model optimizer command ...
        model_fpath = Path(onnx_output + '/' + 'model.onnx')
        ir_data_type = "FP32"
        input_ids = 'input_ids[{} {}]'.format(config.batch_size, config.sequence_length)
        attention_mask = 'attention_mask[{} {}]'.format(config.batch_size, config.sequence_length)
        token_type_ids = 'token_type_ids[{} {}]'.format(config.batch_size, config.sequence_length)
        decoder_input_ids = 'decoder_input_ids[{} {}]'.format(config.batch_size, 1)
        decoder_attention_mask = 'decoder_attention_mask[{} {}]'.format(config.batch_size, 1)

        if config.model in {"gpt2", "facebook/bart-large-cnn", "roberta-base", "xlnet-base-cased", "prophetnet-large-uncased", "distilbert-base-uncased"}:
            input_names = '"' + input_ids + ',' + attention_mask + '"'
        elif config.model in {"t5-small"}:
            input_names = '"' + input_ids + ',' + attention_mask + ',' + decoder_input_ids + ',' + decoder_attention_mask + '"'
        else:
            input_names = '"' + input_ids + ',' + token_type_ids + ',' + attention_mask + '"'

        mo_cmd = f"mo_onnx.py \
            --input_model {model_fpath} \
            --data_type {ir_data_type} \
            --input {input_names} \
            --output_dir {output_ov}  \
            --model_name {model}"
        print(mo_cmd)

        results = subprocess.check_output(mo_cmd, shell=True)

        return model_path

    @classmethod
    def allocate(cls, config: 'BenchmarkConfig'):
        model_name = config.model
        try:
            model_name = model_name.split('/')[1]
        except:
            pass
        onnx_model_path = Path(f"{ONNX_GRAPHS_FOLDER}/{config.model}.{getpid()}.onnx")
        ov_path = Path(f"{ONNX_GRAPHS_FOLDER}/{model_name}.{getpid()}.ov")
        ov_model_path = OpenVINORuntimeBackend.convert(config, model_name, onnx_model_path, ov_path, config.backend.opset)

        backend = OpenVINORuntimeBackend(config.model, onnx_model_path.absolute().as_posix(), ov_model_path.absolute().as_posix())
        backend.configure(config.backend)
        return backend

    def configure(self, config: OpenVINORuntimeConfig):
        super().configure(config)

        LOGGER.info("Configuring OpenVINO Benchmark:")
        self.config['ENFORCE_BF16'] = 'NO'

        if config.num_threads is not None:
            self.config['CPU_THREADS_NUM'] = str(config.num_threads)

            LOGGER.info(f"\t- Setting num_threads({self.config['CPU_THREADS_NUM']})")

        if config.num_streams is not None:
            self.config['CPU_THROUGHPUT_STREAMS'] = str(config.num_streams)

            LOGGER.info(f"\t- Setting num_streams({self.config['CPU_THROUGHPUT_STREAMS']})")


    def execute(self, config: 'BenchmarkConfig', is_reference: bool = False) -> Tuple[Benchmark, np.ndarray]:
        benchmark = Benchmark()

        model_path = Path(self.ov_path)
        model_xml = str(model_path) + '.xml'
        model_bin = str(model_path) + '.bin'

        # Run inference
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        net.batch_size = int(config.batch_size)
        exec_net = ie.load_network(network=net, device_name=config.device.upper(), config=self.config)

        input_name = next(iter(net.inputs))
        output_name = next(iter(net.outputs))

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )
        
        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.NUMPY,
        )

        input_seqs = {k: v.astype("i8") for k, v in inputs.items()}
        
        # Warmup
        outputs = []
        for _ in trange(config.warmup_runs, desc="Warming up"):
            output = exec_net.infer(inputs=input_seqs)
            outputs.append(output[output_name])

        # Let's not run the benchmark for the reference backend,
        # as we are more interested in the output tensors.
        if not is_reference:

            # Run benchmark
            benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
            while sum(benchmark.latencies) < benchmark_duration_ns:
                with benchmark.track():
                    exec_net.infer(inputs=input_seqs)

            benchmark.finalize(benchmark_duration_ns)
        return benchmark, np.stack(outputs)

    def clean(self, config: 'BenchmarkConfig'):
        onnx_path = str(self.onnx_path).split('.onnx')[0]
        model_fpath = Path(onnx_path + '/' + 'model.onnx')
        model_path = Path(self.ov_path)
        model_xml = Path(str(model_path) + '.xml')
        model_bin = Path(str(model_path) + '.bin')
        model_mapping = Path(str(model_path) + '.mapping')

        model_fpath.unlink()
        model_xml.unlink()
        model_bin.unlink()
        model_mapping.unlink()
