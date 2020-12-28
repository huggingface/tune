from typing import Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_class
from omegaconf import OmegaConf

from backends import Backend
from backends.pytorch import PyTorchConfig
from backends.tensorflow import TensorflowConfig
from config import BenchmarkConfig

# Register configurations
cs = ConfigStore.instance()
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="torchscript", node=PyTorchConfig)
cs.store(group="backend", name="tensorflow", node=TensorflowConfig)
cs.store(group="backend", name="xla", node=TensorflowConfig)


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    print(OmegaConf.to_yaml(config))

    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend = backend_factory.allocate(config)
    benchmark = backend.execute(config)

    # Export the result
    df = benchmark.to_pandas()
    df.to_csv("results.csv", index_label="run")


if __name__ == '__main__':
    run()