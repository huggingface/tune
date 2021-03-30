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
from logging import getLogger
from typing import Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_class

from backends import Backend
from backends.ort import OnnxRuntimeConfig
from backends.pytorch import PyTorchConfig
from backends.tensorflow import TensorflowConfig
from config import BenchmarkConfig

# Register configurations
cs = ConfigStore.instance()
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backend", name="torchscript_backend", node=PyTorchConfig)
cs.store(group="backend", name="tensorflow_backend", node=TensorflowConfig)
cs.store(group="backend", name="xla_backend", node=TensorflowConfig)
cs.store(group="backend", name="ort_backend", node=OnnxRuntimeConfig)


LOGGER = getLogger("benchmark")


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend = backend_factory.allocate(config)
    benchmark = backend.execute(config)
    backend.clean(config)

    df = benchmark.to_pandas()
    df.to_csv("results.csv", index_label="id")


if __name__ == '__main__':
    run()
