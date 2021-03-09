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
from logging import getLogger, DEBUG, INFO
from logging.handlers import QueueHandler
from multiprocessing.connection import Connection
from multiprocessing import Queue
from os import environ
from threading import Thread
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


LOGGER = getLogger("benchmark")


def logging_thread(log_queue: Queue):
    while True:
        record = log_queue.get()
        if record is None:
            break

        getLogger(record.name).handle(record)


def allocate_and_run_model(config: BenchmarkConfig, socket_binding: List[int], core_binding: List[int], pipe_out: Connection, log_queue: Queue):
    from hydra.utils import get_class
    from utils import MANAGED_ENV_VARIABLES, configure_numa

    # Setup logging to log records in the same file descriptor than main process
    qh = QueueHandler(log_queue)
    root = getLogger()
    root.setLevel(DEBUG if hasattr(config, "debug") and config.debug else INFO)
    root.addHandler(qh)

    # Out of the box setup shouldn't set any NUMA affinity
    if config.openmp is not None and config.openmp.name != "none":
        # Configure CPU threads affinity for current process
        configure_numa(socket_binding, core_binding)
    else:
        LOGGER.debug(f"Skipping configuring NUMA, config.openmp = {config.openmp.name}")

    # Print LD_PRELOAD information to ensure it has been correctly setup by launcher
    env_to_print = environ.keys() if hasattr(config, "debug") and config.debug else MANAGED_ENV_VARIABLES
    for env_var in env_to_print:
        LOGGER.info(f"[ENV] {env_var}: {environ.get(env_var)}")

    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend = backend_factory.allocate(config)
    benchmark = backend.execute(config)
    backend.clean(config)

    # Write out the result to the pipe
    pipe_out.send(benchmark)


@hydra.main(config_path="../configs", config_name="benchmark")
def run(config: BenchmarkConfig) -> None:
    # TODO: Check why imports are not persisted when doing swaps
    from multiprocessing import Pipe, Process, set_start_method
    from utils import get_instances_with_cpu_binding, set_ld_preload_hook

    set_start_method("spawn")

    # Configure eventual additional LD_PRELOAD
    set_ld_preload_hook(config)

    # Get the set of threads affinity for this specific process
    instance_core_bindings = get_instances_with_cpu_binding(config.num_core_per_instance, config.num_instances)

    if hasattr(config, "debug") and config.debug:
        environ["OMP_DISPLAY_ENV"] = "VERBOSE"
        environ["KMP_SETTINGS"] = "1"

    if config.num_instances > 1:
        LOGGER.info(f"Starting Multi-Instance inference setup: {config.num_instances} instances")

    # Allocate all the IPC tools
    (reader, writer), log_queue = Pipe(duplex=False), Queue()

    # Allocate log record listener thread
    log_thread = Thread(target=logging_thread, args=(log_queue, ))
    log_thread.start()

    # Allocate model and run benchmark
    benchmarks, workers = [], []
    for allocated_socket, allocated_socket_cores in instance_core_bindings:
        process = Process(
            target=allocate_and_run_model,
            kwargs={
                "config": config,
                "socket_binding": allocated_socket,
                "core_binding": allocated_socket_cores,
                "pipe_out": writer,
                "log_queue": log_queue
            }
        )
        process.start()
        LOGGER.debug(f"Started process with pid {process.pid}")
        workers.append(process)

    # Wait for all workers
    for worker in workers:
        worker.join()
        benchmark = reader.recv()
        benchmarks.append(benchmark)

    # Stop logger thread
    log_queue.put_nowait(None)
    log_thread.join()

    # Export the result
    if len(benchmarks) > 1:
        benchmark = Benchmark.merge(benchmarks)
    else:
        benchmark = benchmarks[0]

    df = benchmark.to_pandas()
    df.to_csv("results.csv", index_label="id")


if __name__ == '__main__':
    run()
