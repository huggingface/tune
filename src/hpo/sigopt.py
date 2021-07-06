#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

# import subprocess
# import sys
import copy
# from glob import glob
# from pathlib import Path
# from functools import reduce
import json
import os
from binascii import hexlify
from collections import OrderedDict
# from consolidate import gather_results, aggregate_multi_instances_results, SCALING_CHOICES
from random import getrandbits

from sigopt import Connection

# from enum import IntEnum
# from typing import Any, Dict, NamedTuple
# from argparse import ArgumentParser
from utils.cpu import CPUinfo

from .utils import (TuningMode, check_tune_function_kwargs,
                    generate_nb_cores_candidates,
                    generate_nb_instances_candidates, launch_and_wait)


def convert_fmt(file):
    target = os.path.splitext(file)[0] + ".csv"
    with open(file) as f, open(target, "w") as new_f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.rstrip())
            if i == 0:
                new_f.write(",".join(line.keys()) + "\n")
            new_f.write(",".join([str(v) for v in line.values()]) + "\n")


# def launch_and_wait(parameters: Dict[str, Any]):
#     args.experiment_id = hexlify(getrandbits(32).to_bytes(4, 'big')).decode('ascii')
#     cmd = [sys.executable, "launcher.py", f"--experiment_id={args.experiment_id}"]
#
#     for name, value in parameters.items():
#         # Number of cores
#         if name == "nb_cores":
#             cmd.append(f"--ncore_per_instance={value}")
#
#         # Multi instances
#         if name == "ninstances":
#             cmd += ["--multi_instance", f"--ninstances={value}"] if value > 1 else ["--ninstances=1"]
#
#         # OpenMP
#         elif name == "openmp" and value == "iomp":
#             cmd.append("--enable_iomp")
#
#         # Memory allocator
#         elif name == "allocator":
#             if value == "default":
#                 cmd.append("--use_default_allocator")
#             else:
#                 cmd.append(f"--enable_{value}")
#
#         # Transparent huge pages
#         elif name == "huge_pages" and value.lower() == "on":
#             cmd.append("--enable_thp")
#
#         # kmp_blocktime
#         elif name == "kmp_blocktime":
#             cmd.append(f"--kmp_blocktime={value}")
#
#     # numactl
#     if args.disable_numactl:
#         cmd.append("--disable_numactl")
#
#     bs = args.batch_size if args.mode != TuningMode.LATENCY else args.batch_size // parameters["ninstances"]
#
#     # Main script parameter
#     cmd += [
#         "--",
#         "src/main.py",
#         "benchmark_duration=60",
#         "warmup_runs=5",
#         f"batch_size={bs}",
#         f"sequence_length={args.sequence_length}",
#         f"backend={args.framework}"
#     ]
#
#     print("----- launcher cmd: [%s]" % " ".join(cmd), flush=True)
#     process = subprocess.Popen(cmd, cwd=os.curdir, stdout=subprocess.PIPE)
#     process.wait()
#     output = process.stdout.read().decode("utf-8")
#
#     scaling_choices = ["batch-size-scaling", "core-count-scaling"]
#     latency, throughput = parse_results(args.experiment_id, scaling_choices[args.mode >= TuningMode.THROUGHPUT])
#
#     # return ExperimentResult(latency, throughput)
#     return [{'name': 'latency', 'value': latency}, {'name': 'throughput', 'value': throughput}]


# def parse_results(experiment_id, multi_instances_scaling=None):
#     results_folder = Path("outputs/default/" + experiment_id)
#     try:
#         # Detect folder run type from folder structure
#         instances_folder = glob(f"{results_folder.as_posix()}/*")
#         is_multi_instances = len(instances_folder) > 1
#
#         # If we detect multi instance and no scaling mode is provided, ask for a value
#         if is_multi_instances and multi_instances_scaling is None:
#             print("warning: multi_instances_scaling is not specified for multi-instance run.", flush=True)
#
#         # Gather the results to manipulate
#         consolidated_df, sorting_columns = gather_results(results_folder, is_multi_instances)
#
#         if is_multi_instances and multi_instances_scaling is not None:
#             agg_df = aggregate_multi_instances_results(consolidated_df, sorting_columns, multi_instances_scaling)
#             latency = float(agg_df['latency_mean']['mean'].iloc[0] / 1e6)
#             throughput = float(agg_df['throughput']['sum'].iloc[0])
#         else:
#             latency = float(consolidated_df['latency_mean'].iloc[0] / 1e6)
#             throughput = float(consolidated_df['throughput'].iloc[0])
#
#     except ValueError as ve:
#         print(ve, flush=True)
#         latency, throughput = None, None
#
#     return latency, throughput


def create_experiment(conn, experiment_info):
    cpu_info = CPUinfo()
    metrics = {
        TuningMode.LATENCY: [{"name": "latency", "objective": "minimize"}],
        TuningMode.THROUGHPUT: [{"name": "throughput", "objective": "maximize"}],
        TuningMode.BOTH: [
            {"name": "latency", "objective": "minimize"},
            {"name": "throughput", "objective": "maximize"},
        ],
    }

    name = experiment_info["name"]
    mode = experiment_info["mode"]
    batch_size = experiment_info["batch_size"]
    sequence_length = experiment_info["sequence_length"]

    # If a project name is specified in the experiment_info, use it, else check if present as an
    # environment variable.
    project = experiment_info.get("project")
    if project is None:
        project = os.environ.get("SIGOPT_PROJECT")
    if project is None:
        raise ValueError("a sigopt project name needs to be specified")

    experiment_meta = {
        "name": f"{name}-mode-{mode.value}-bs-{batch_size}-seq-{sequence_length}",
        "project": project,
        "parameters": [
            {
                "categorical_values": ["openmp", "iomp"],
                "name": "openmp",
                "type": "categorical",
            },
            {
                "categorical_values": ["default", "tcmalloc", "jemalloc"],
                "name": "allocator",
                "type": "categorical",
            },
            {
                "categorical_values": ["on", "off"],
                "name": "huge_pages",
                "type": "categorical",
            },
            {"bounds": {"min": 0, "max": 1}, "name": "kmp_blocktime", "type": "int"},
        ],
        "observation_budget": experiment_info["n_trials"],
        "metrics": metrics[mode],
        "parallel_bandwidth": 1,
    }

    candidate_type, nb_cores = generate_nb_cores_candidates(batch_size, mode, cpu_info)
    if candidate_type == "range":
        experiment_meta["parameters"].append(
            {
                "bounds": {"min": nb_cores[0], "max": nb_cores[1]},
                "name": "nb_cores",
                "type": "int",
            }
        )
    if candidate_type == "discrete" and len(nb_cores) > 1:
        nb_cores = list(map(str, nb_cores))
        experiment_meta["parameters"].append(
            {"categorical_values": nb_cores, "name": "nb_cores", "type": "categorical"}
        )

    experiment = conn.experiments().create(**experiment_meta)
    print(
        "created experiment: https://app.sigopt.com/experiment/" + experiment.id,
        flush=True,
    )
    return experiment


def sigopt_tune(launcher_parameters=None, main_parameters=None, **kwargs):
    check_tune_function_kwargs(kwargs)
    # TODO: check sigopt specific kwargs that need to be specified.
    kwargs = copy.deepcopy(kwargs)

    cpu_info = CPUinfo()

    if kwargs.get("convert_csv", False):
        if "logfile" not in kwargs:
            raise ValueError(
                f"a log file needs to be specified when convert_csv is True"
            )
        convert_fmt(kwargs["logfile"])
        return

    conn = Connection()
    if "proxy" in kwargs:
        conn.set_proxies({"http": kwargs["proxy"]})

    mode = kwargs["mode"]

    # TODO: is it better to have everything explicitly set or just to pass kwargs as
    # experiment_info?
    experiment_info = {
        "name": kwargs["exp_name"],
        "mode": kwargs["mode"],
        "n_trials": kwargs["n_trials"],
        "batch_size": main_parameters["batch_size"],
        "sequence_length": main_parameters["sequence_length"],
    }

    project_name = kwargs.get("project", None)
    if project_name:
        experiment_info["project"] = project_name

    experiment = create_experiment(conn, experiment_info)
    experiment_id = hexlify(getrandbits(32).to_bytes(4, "big")).decode("ascii")

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments_dict = dict(suggestion.assignments)
        if "nb_cores" not in assignments_dict.keys():
            assignments_dict["nb_cores"] = cpu_info.physical_core_nums

        # Using dummy value for batch size (1) as it will not be used since the number of cores is
        # provided.
        assignments_dict["ninstances"] = generate_nb_instances_candidates(
            1, mode, cpu_info, nb_cores=int(assignments_dict["nb_cores"])
        )

        exp_result = launch_and_wait(
            launcher_parameters=assignments_dict,
            main_parameters=main_parameters,
            experiment_id=experiment_id,
        )
        values = {
            TuningMode.LATENCY: {"name": "latency", "value": exp_result.latency},
            TuningMode.THROUGHPUT: {
                "name": "throughput",
                "value": exp_result.throughput,
            },
            TuningMode.BOTH: [
                {"name": "latency", "value": exp_result.latency},
                {"name": "throughput", "value": exp_result.throughput},
            ],
        }
        print(
            "suggestion: %s, values: %s" % (suggestion.assignments, values), flush=True
        )
        if (
            values[TuningMode.LATENCY]["value"] is None
            and values[TuningMode.THROUGHPUT]["value"] is None
        ):
            conn.experiments(experiment.id).suggestions(suggestion.id).delete()
            print(
                "incomplete metrics for suggestion: %s" % suggestion.assignments,
                flush=True,
            )
        else:
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                values=values[mode],
            )
        experiment = conn.experiments(experiment.id).fetch()

    # Dump the best result into file
    best_assignment = list(
        conn.experiments(experiment.id).best_assignments().fetch().iterate_pages()
    )[0]
    report = OrderedDict(best_assignment.assignments)

    if "nb_cores" not in report.keys():
        report["nb_cores"] = cpu_info.physical_core_nums

    report = OrderedDict(sorted(report.items()))

    report["metrics_name"] = best_assignment.values[0].name
    report["metrics_value"] = best_assignment.values[0].value
    report["batch_size"] = main_parameters["batch_size"]
    report["sequence_length"] = main_parameters["sequence_length"]
    report["mode"] = kwargs["mode"].value
    report["framework"] = main_parameters["backend"]
    report["experiment_id"] = experiment_id

    with open(kwargs["logfile"], "a") as f:
        json.dump(report, f)
        f.write("\n")
