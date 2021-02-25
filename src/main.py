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
from typing import Type, List

import hydra
from hydra.core.config_store import ConfigStore

from backends import Backend
from backends.ort import OnnxRuntimeConfig
from backends.pytorch import PyTorchConfig
from backends.tensorflow import TensorflowConfig
from benchmark import Benchmark
from config import BenchmarkConfig


# Register configurations
cs = ConfigStore.instance()
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backend", name="torchscript_backend", node=PyTorchConfig)
cs.store(group="backend", name="tensorflow_backend", node=TensorflowConfig)
cs.store(group="backend", name="xla_backend", node=TensorflowConfig)
cs.store(group="backend", name="ort_backend", node=OnnxRuntimeConfig)


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    # TODO: Check why imports are not persisted when doing swaps
    from logging import getLogger
    from multiprocessing import Pipe
    from multiprocessing.connection import Connection
    from multiprocessing.context import Process
    from os import environ, system, getpid

    from hydra.utils import get_class
    from utils import MANAGED_ENV_VARIABLES, get_instances_with_cpu_binding

    LOGGER = getLogger("benchmark")

    if hasattr(config, "malloc") and "tcmalloc" == config.malloc.name:
        from utils import check_tcmalloc
        check_tcmalloc()

    # Print LD_PRELOAD information to ensure it has been correctly setup by launcher
    for env_var in MANAGED_ENV_VARIABLES:
        LOGGER.info(f"[ENV] {env_var}: {environ.get(env_var)}")

    def allocate_and_run_model(core_binding: List[int], pipe_out: Connection):
        # Configure CPU threads affinity for current process
        system(f"taskset -p -c {','.join(map(str, core_binding))} {getpid()}")

        LOGGER.info(f"[TASKSET] Set CPU affinity to: {core_binding} (pid={getpid()})")

        backend_factory: Type[Backend] = get_class(config.backend._target_)
        backend = backend_factory.allocate(config)
        benchmark = backend.execute(config)
        backend.clean(config)

        # Write out the result to the pipe
        pipe_out.send(benchmark)

    # Get the set of threads affinity for this specific process
    instance_core_bindings = get_instances_with_cpu_binding(config.num_threads_per_instance, config.num_instances)

    if len(instance_core_bindings) > 0:
        LOGGER.info(f"Starting Multi-Instance inference setup: {len(instance_core_bindings)} instances")

    # Allocate all the model instances
    reader, writer = Pipe(duplex=False)
    benchmarks, workers = [], []
    for instance_core_binding in instance_core_bindings:
        process = Process(
            target=allocate_and_run_model,
            kwargs={
                "core_binding": instance_core_binding,
                "pipe_out": writer
            }
        )
        process.start()
        workers.append(process)

    # Wait for all workers
    for worker in workers:
        worker.join()
        benchmark = reader.recv()

        benchmarks.append(benchmark)

    # Export the result
    if len(benchmarks) > 1:
        benchmark = Benchmark.merge(benchmarks)
    else:
        benchmark = benchmarks[0]

    df = benchmark.to_pandas()
    df.to_csv("results.csv", index_label="id")


if __name__ == '__main__':
    run()
