#  Copyright 2021 Intel Corporation.
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

import itertools

from argparse import ArgumentParser
from hpo.optuna import optuna_tune
from hpo.sigopt import sigopt_tune
from hpo.utils import TuningMode

# Predefined backend lists; Please change as needed

BACKEND_SET_CHOICES = {"pt", "tf", "pcl", "bert_cpp"}
pt_experiment_backends = ["pytorch", "torchscript"]
tf_experiment_backends = ["tensorflow", "xla"]
pcl_experiment_backends = ["pcl", "pytorch", "torchscript"]
bert_cpp_experiment_backends = ["fused", "pytorch", "torchscript"]

# Default command settings

batch_size_all = [1, 4, 8, 16, 32]
sequence_length_all = [20, 32, 128, 384, 512]
benchmark_duration_default = [60]
benchmark_duration_long = [120]
benchmark_duration_short = [10]
warmup_runs_default = [5]
command_prefix = "PYTHONPATH=src python3"
main_prefix = "-- src/main.py --multirun"
launcher_prefix = "launcher.py --multi_instance"
oob_command_prefix = "PYTHONPATH=src python3 -- src/main.py --multirun"
launcher_command_prefix = "PYTHONPATH=src python3 launcher.py --multi_instance"
kmp_affinity_default = "verbose,granularity=fine,compact,1,0"
backends = pt_experiment_backends
backend_specific_knobs = ""
tf_specific_knobs = "backend.num_interops_threads=1"
enable_iomp_default = [True, False]
malloc_default = ["use_default_allocator", "enable_tcmalloc", "enable_jemalloc"]

# Read in command line args

parser = ArgumentParser("Hugging Face Model Benchmarking")
parser.add_argument(
    "--dryrun",
    action="store_true",
    help="Prints out only the command lines and does not execute",
)
parser.add_argument(
    "--backend-list",
    choices=BACKEND_SET_CHOICES,
    help="Select predetermined backend list",
)
parser.add_argument(
    "--mode",
    type=TuningMode,
    choices=[
        TuningMode.LATENCY,
        TuningMode.THROUGHPUT,
        TuningMode.BOTH,
    ],
    default=TuningMode.BOTH,
    help="The criteria to use for the benchmark",
)
args = parser.parse_args()
if args.backend_list == "pt":
    backends = pt_experiment_backends
if args.backend_list == "tf":
    backends = tf_experiment_backends
    backend_specific_knobs = tf_specific_knobs
if args.backend_list == "pcl":
    backends = pcl_experiment_backends
if args.backend_list == "bert_cpp":
    backends = bert_cpp_experiment_backends

# Define the experiments

oob_experiments = [
    {
        "name": "oob_experiments",
        "launcher_knobs": {},
        "main_knobs": {
            "backend": backends,
            "batch_size": batch_size_all,
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    }
]

bs1_experiments = [
    {
        "name": "bs1_experiments",
        "launcher_knobs": {
            "ninstances": [1, 2, 4, 5, 10, 20, 40, 80],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [1],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    }
]

bs4_batch_size_scaling_experiments = [
    {
        "name": "bs4_experiments_bss_inst1",
        "launcher_knobs": {
            "ninstances": [1],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [4],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs4_experiments_bss_inst2",
        "launcher_knobs": {
            "ninstances": [2],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [2],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs4_experiments_bss_inst4",
        "launcher_knobs": {
            "ninstances": [4],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [1],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
]

bs8_batch_size_scaling_experiments = [
    {
        "name": "bs8_experiments_bss_inst1",
        "launcher_knobs": {
            "ninstances": [1],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [8],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs8_experiments_bss_inst2",
        "launcher_knobs": {
            "ninstances": [2],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [4],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs8_experiments_bss_inst4",
        "launcher_knobs": {
            "ninstances": [4],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [2],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs8_experiments_bss_inst8",
        "launcher_knobs": {
            "ninstances": [8],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [1],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
]

bs16_batch_size_scaling_experiments = [
    {
        "name": "bs16_experiments_bss_inst1",
        "launcher_knobs": {
            "ninstances": [1],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [16],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs16_experiments_bss_inst2",
        "launcher_knobs": {
            "ninstances": [2],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [8],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs16_experiments_bss_inst4",
        "launcher_knobs": {
            "ninstances": [4],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [4],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs16_experiments_bss_inst8",
        "launcher_knobs": {
            "ninstances": [8],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [2],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs16_experiments_bss_inst16",
        "launcher_knobs": {
            "ninstances": [16],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [1],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
]

bs32_batch_size_scaling_experiments = [
    {
        "name": "bs32_experiments_bss_inst1",
        "launcher_knobs": {
            "ninstances": [1],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [32],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs32_experiments_bss_inst2",
        "launcher_knobs": {
            "ninstances": [2],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [16],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs32_experiments_bss_inst4",
        "launcher_knobs": {
            "ninstances": [4],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [8],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs32_experiments_bss_inst8",
        "launcher_knobs": {
            "ninstances": [8],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [4],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
    {
        "name": "bs32_experiments_bss_inst16",
        "launcher_knobs": {
            "ninstances": [16],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [2],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    },
]

core_count_scaling_experiments = [
    {
        "name": "ccs_experiments",
        "launcher_knobs": {
            "ninstances": [1, 2, 4, 5, 10, 20],
            "kmp_affinity": kmp_affinity_default,
            "enable_iomp": enable_iomp_default,
            "malloc": malloc_default,
        },
        "main_knobs": {
            "backend": backends,
            "batch_size": [4, 8, 16, 32],
            "sequence_length": sequence_length_all,
            "benchmark_duration": benchmark_duration_default,
            "warmup_runs": warmup_runs_default,
        },
    }
]

experiment_list = [
    # oob_experiments,
    bs1_experiments,
    bs4_batch_size_scaling_experiments,
    bs8_batch_size_scaling_experiments,
    bs16_batch_size_scaling_experiments,
    bs32_batch_size_scaling_experiments,
    core_count_scaling_experiments,
]


if __name__ == "__main__":
    key_mapping = {
        "ninstances": "instances",
        "enable_iomp": "openmp",
        "malloc": "allocator",
    }

    for experiments in experiment_list:
        for experiment in experiments:
            print("-" * 40)
            print(f"Running experiment {experiment['name']}")
            print("-" * 40)

            # Processing the parameter values to match with what the hpo function expects.
            launcher_parameters = dict(experiment["launcher_knobs"])
            for orig_key, mapped_key in key_mapping.items():
                if orig_key in launcher_parameters:
                    launcher_parameters[mapped_key] = launcher_parameters[orig_key]
                    del launcher_parameters[orig_key]

            if isinstance(launcher_parameters.get("openmp", None), list):
                map_enable_iomp = {
                    (True,): "iomp",
                    (False,): "openmp",
                    (True, False): ["openmp", "iomp"],
                }
                launcher_parameters["openmp"] = map_enable_iomp.get(
                    tuple(launcher_parameters["openmp"]), launcher_parameters["openmp"]
                )

            allocator = launcher_parameters.get("allocator", None)
            if allocator is not None:
                map_allocator = {
                    "use_default_allocator": "default",
                    "enable_tcmalloc": "tcmalloc",
                    "enable_jemalloc": "jemalloc",
                }
                allocator = launcher_parameters.get("allocator", None)
                if isinstance(allocator, list):
                    launcher_parameters["allocator"] = [
                        map_allocator.get(name, name) for name in allocator
                    ]
                if isinstance(allocator, str):
                    launcher_parameters["allocator"] = map_allocator.get(
                        allocator, allocator
                    )

            # "Flattening" for multiple batch_size / sequence_length for now as --multirun support is
            # not well tested for inference / throughput measurement.
            main_parameters = experiment["main_knobs"]

            batch_size = main_parameters["batch_size"]
            if isinstance(batch_size, int):
                batch_size = [batch_size]
            batch_size_is_fixed = len(batch_size) == 1

            sequence_length = main_parameters["sequence_length"]
            if isinstance(sequence_length, int):
                sequence_length = [sequence_length]
            sequence_length_is_fixed = len(sequence_length) == 1

            backend = main_parameters["backend"]
            if isinstance(backend, str):
                backend = [backend]
            backend_is_fixed = len(backend) == 1

            product = itertools.product(batch_size, sequence_length, backend)

            for (bs, seqlen, back) in product:
                exp_name = f"{experiment['name']}"
                if not batch_size_is_fixed:
                    exp_name = f"{exp_name}_bs{bs}"
                if not sequence_length_is_fixed:
                    exp_name = f"{exp_name}_seqlen{seqlen}"
                if not backend_is_fixed:
                    exp_name = f"{exp_name}_{back}"

                main_parameters["batch_size"] = bs
                main_parameters["sequence_length"] = seqlen
                main_parameters["backend"] = back

                print("\nTuning with Optuna")
                print("-" * 40)
                print("\n")

                optuna_tune(
                    launcher_parameters=launcher_parameters,
                    main_parameters=main_parameters,
                    exp_name=exp_name,
                    mode=args.mode,
                    n_trials=50,
                )

                print("\nTuning with Sigopt")
                print("-" * 40)
                print("\n")

                sigopt_tune(
                    launcher_parameters=launcher_parameters,
                    main_parameters=main_parameters,
                    exp_name=exp_name,
                    mode=args.mode,
                    n_trials=50,
                )
