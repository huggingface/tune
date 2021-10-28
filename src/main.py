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
from typing import Type, get_args, Union

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.experimental import compose
from hydra.utils import get_class
from omegaconf import OmegaConf, DictConfig

from backends import Backend, BackendConfig
from backends.ort import OnnxRuntimeConfig
from backends.ov import OpenVINORuntimeConfig
from backends.pytorch import PyTorchConfig
from backends.tensorflow import TensorflowConfig
from config import BenchmarkConfig


# Register resolvers
OmegaConf.register_new_resolver("pytorch_version", PyTorchConfig.version)
OmegaConf.register_new_resolver("tensorflow_version", TensorflowConfig.version)
OmegaConf.register_new_resolver("ort_version", OnnxRuntimeConfig.version)
OmegaConf.register_new_resolver("ov_version", OpenVINORuntimeConfig.version)

# Register configurations
cs = ConfigStore.instance()
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backend", name="torchscript_backend", node=PyTorchConfig)
cs.store(group="backend", name="tensorflow_backend", node=TensorflowConfig)
cs.store(group="backend", name="tensorflow_graph_backend", node=TensorflowConfig)
cs.store(group="backend", name="ort_backend", node=OnnxRuntimeConfig)
cs.store(group="backend", name="ov_backend", node=OpenVINORuntimeConfig)


LOGGER = getLogger("benchmark")


def get_overrided_backend_config(original_config: Union[DictConfig, BackendConfig], override: str) -> DictConfig:
    # Copy the initial config and pop the backend
    update_config = original_config.copy()
    OmegaConf.set_struct(update_config, False)
    update_config.pop("backend")

    # Retrieve the original backend factory
    backend_factory: Type[Backend] = get_class(original_config.backend._target_)

    # Compose the two configs (reference <- original @backend==config.reference)
    reference_config = compose(config_name="benchmark", overrides=[f"backend={override}"])
    reference_config.merge_with(update_config)
    reference_backend_factory: Type[Backend] = get_class(reference_config.backend._target_)

    # Retrieve each original & reference BackendConfig instance type
    reference_backend_config_type: Type[BackendConfig] = get_args(reference_backend_factory.__orig_bases__[0])[0]
    original_backend_config_type: Type[BackendConfig] = get_args(backend_factory.__orig_bases__[0])[0]

    # Filter out to rely only on the common subset of supported config elements
    reference_backend_keys = reference_backend_config_type.supported_keys()
    original_backend_keys = original_backend_config_type.supported_keys()

    # (A - B) union (A inter B)
    overlapping_backend_config_keys = \
        (reference_backend_keys.intersection(original_backend_keys)) - {"name", "_target_", "version"}

    LOGGER.debug(f"Keys to override from original config in the new one: {overlapping_backend_config_keys}")

    # Get a masked configuration copy
    original_overlapping_backend_config = OmegaConf.masked_copy(
        original_config,
        list(overlapping_backend_config_keys)
    )

    # Override the properties
    reference_config["backend"].merge_with(original_overlapping_backend_config)

    return reference_config


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    # We need to allocate the reference backend (used to compare backend output against)
    if config.reference is not None and config.reference != config.backend:
        LOGGER.info(f"Using {config.reference} as reference backend")
        reference_config = get_overrided_backend_config(config, override=config.reference)
    else:
        reference_config = None

    # Allocate requested target backend
    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend = backend_factory.allocate(config)

    # Run benchmark and reference
    benchmark, outputs = backend.execute(config, is_reference=False)
    backend.clean(config)

    if reference_config is not None:
        reference_backend_factory = get_class(reference_config.backend._target_)
        reference_backend = reference_backend_factory.allocate(reference_config)
        _, ref_outputs = reference_backend.execute(reference_config, is_reference=True)

        # Record the outputs to compare with the target backend
        benchmark.record_outputs(outputs, ref_outputs)
        reference_backend.clean(reference_config)

        LOGGER.info(
            f"Reference backend ({config.reference}) against target backend ({config.backend.name})"
            f" absolute difference:"
            f" {np.mean(benchmark.outputs_diff)} (+/- {np.std(benchmark.outputs_diff)})"
            f" over {len(benchmark.outputs_diff)} sample(s)"
        )

    # Save the resolved config
    OmegaConf.save(config, ".hydra/config.yaml", resolve=True)

    df = benchmark.to_pandas()
    df['model'] = model_name
    df['backend'] = config.backend.name
    df['seq_len'] = config.sequence_length
    df['batch_size'] = config.batch_size
    df['num_threads'] = config.backend.num_threads
    df.to_csv("{}-{}-results.csv".format(config.backend.name, model_name), index_label="id")


if __name__ == '__main__':
    run()
