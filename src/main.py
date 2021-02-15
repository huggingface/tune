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

from os import system
from os.path import exists
from typing import Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_class
from omegaconf import OmegaConf

from backends import Backend
from backends.ort import OnnxRuntimeConfig
from backends.pytorch import PyTorchConfig
from backends.tensorflow import TensorflowConfig
from config import BenchmarkConfig

# Transparent Huge Pages default location
DEBIAN_TRANSPARENT_PAGES_PATH = "/sys/kernel/mm/transparent_hugepage/enabled"
REDHAT_TRANSPARENT_PAGES_PATH = "/sys/kernel/mm/redhat_transparent_hugepage/enabled"

# Register configurations
cs = ConfigStore.instance()
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="torchscript", node=PyTorchConfig)
cs.store(group="backend", name="tensorflow", node=TensorflowConfig)
cs.store(group="backend", name="xla", node=TensorflowConfig)
cs.store(group="backend", name="ort", node=OnnxRuntimeConfig)


def use_transparent_huge_page(enable: bool):
    # Look for common TBH places
    if exists(DEBIAN_TRANSPARENT_PAGES_PATH):
        tbh_config_path = DEBIAN_TRANSPARENT_PAGES_PATH
    elif exists(REDHAT_TRANSPARENT_PAGES_PATH):
        tbh_config_path = REDHAT_TRANSPARENT_PAGES_PATH
    else:
        print("Unable to locate Transparent Huge Page configuration files on your system, no action will be taken.")
        return

    print(f"Found Transparent Huge Page configuration files at: {tbh_config_path}")

    # Turn on (always) / off (never) TBH
    thb_status = "always" if enable else "never"
    system(f"echo {thb_status} > {tbh_config_path}")
    print(f"Transparent Huge Page enabled: {system(f'cat {tbh_config_path}')}")


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Turn on/off usage of Transparent Huge Pages
    use_transparent_huge_page(config.use_huge_page)

    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend = backend_factory.allocate(config)
    benchmark = backend.execute(config)
    backend.clean(config)

    # Export the result
    df = benchmark.to_pandas()
    df.to_csv("results.csv", index_label="run")


if __name__ == '__main__':
    run()