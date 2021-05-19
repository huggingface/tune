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
from binascii import hexlify
from dataclasses import dataclass
from logging import getLogger
from random import getrandbits

from typing import Dict, Optional

from omegaconf import MISSING
from transformers import __version__ as transformers_version

from backends import BackendConfig


LOGGER = getLogger("benchmark")


@dataclass()
class BenchmarkConfig:
    # Python interpreter version
    python_version: str = MISSING

    # Store the transformers version used during the benchmark
    transformers_version: str = transformers_version

    # Number of forward pass to run before recording any performance counters.
    warmup_runs: int = MISSING

    # Duration in seconds the benchmark will collect performance counters
    benchmark_duration: int = MISSING

    # The backend to use for recording timing (pytorch, torchscript, tensorflow, xla, onnxruntime)
    backend: BackendConfig = MISSING

    # Name of the model used for the benchmark
    model: str = MISSING

    # CPU or CUDA device to run inference on
    device: str = MISSING

    # The dtype of the model to run inference with (float32, float16, int8, bfloat16)
    precision: str = MISSING

    # Use Transparent Huge Page mechanism to increase CPU cache hit probability
    use_huge_page: str = MISSING

    # Number of sample given to the model at each forward
    batch_size: int = MISSING

    # The length of the sequence (in tokens) given to the model
    sequence_length: int = MISSING

    # Multi instances settings #
    num_instances: int = MISSING

    # Number of core per instances
    num_core_per_instance: int = MISSING

    # Experiment identifier
    experiment_id: str = hexlify(getrandbits(32).to_bytes(4, 'big')).decode('ascii')

    # Experiment name
    experiment_name: str = "default"

    # Identifier for the current instance. Allow to create specific instance config folder
    instance_id: int = 0

    # Reference backend implementation that will be used to generate reference (output tensors)
    reference: Optional[str] = None
