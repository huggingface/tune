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

import copy
import json
import os
from binascii import hexlify
from collections import OrderedDict
from random import getrandbits

from sigopt import Connection

from utils.cpu import CPUinfo
from .utils import (
    TuningMode,
    generate_nb_cores_candidates,
    generate_nb_instances_candidates,
    launch_and_wait,
    tune_function,
)


def convert_fmt(file):
    target = os.path.splitext(file)[0] + ".csv"
    with open(file) as f, open(target, "w") as new_f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.rstrip())
            if i == 0:
                new_f.write(",".join(line.keys()) + "\n")
            new_f.write(",".join([str(v) for v in line.values()]) + "\n")


def create_experiment(
    conn, experiment_info, specified_candidates=None, specified_parameters=None
):
    if specified_candidates is None:
        specified_candidates = {}

    if specified_parameters is None:
        specified_parameters = {}

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
        "parameters": {
            "openmp": {
                "categorical_values": ["openmp", "iomp"],
                "name": "openmp",
                "type": "categorical",
            },
            "allocator": {
                "categorical_values": ["default", "tcmalloc", "jemalloc"],
                "name": "allocator",
                "type": "categorical",
            },
            "huge_pages": {
                "categorical_values": ["on", "off"],
                "name": "huge_pages",
                "type": "categorical",
            },
            "kmp_blocktime": {
                "bounds": {"min": 0, "max": 1},
                "name": "kmp_blocktime",
                "type": "int",
            },
        },
        "observation_budget": experiment_info["n_trials"],
        "metrics": metrics[mode],
        "parallel_bandwidth": 1,
    }

    # If a launcher parameter was specified manually, there are two possibilities:
    #   1. The specified value is a list, in which case it is considered to be a list of
    #       candidates Sigopt should choose from
    #   2. The specified value is not a list, in which case its value is ignored for now (it will
    #       be taken into account in the sigopt_tune function), but the parameter is removed from
    #       the ones Sigopt should tune on.
    for param in experiment_meta["parameters"]:
        if param in specified_parameters:
            experiment_meta["parameters"].pop(param)

    for param in specified_candidates:
        candidates = list(map(str, specified_candidates[param]))
        experiment_meta["parameters"][param] = {
            "categorical_values": list(map(str, candidates)),
            "name": param,
            "type": "categorical",
        }

    # Because we cannot choose both the number of instances and the number of cores with Sigopt,
    # we set the number of cores as a parameter to tune if the number of instances was not set to
    # be tunable.
    if "instances" not in experiment_meta["parameters"]:
        nb_instances = specified_parameters.get("instances", -1)
        candidates_type, nb_cores = generate_nb_cores_candidates(
            batch_size, mode, cpu_info, nb_instances=nb_instances
        )
        if candidates_type == "range":
            experiment_meta["parameters"]["nb_cores"] = {
                "bounds": {"min": nb_cores[0], "max": nb_cores[1]},
                "name": "nb_cores",
                "type": "int",
            }

        if candidates_type == "discrete" and len(nb_cores) > 1:
            nb_cores = list(map(str, nb_cores))
            experiment_meta["parameters"]["nb_cores"] = {
                "categorical_values": nb_cores,
                "name": "nb_cores",
                "type": "categorical",
            }

    print(experiment_meta["parameters"])
    experiment_meta["parameters"] = list(experiment_meta["parameters"].values())

    experiment = conn.experiments().create(**experiment_meta)
    print(
        f"Created experiment: https://app.sigopt.com/experiment/{experiment.id}",
        flush=True,
    )
    return experiment


@tune_function
def sigopt_tune(launcher_parameters=None, main_parameters=None, **kwargs):
    if launcher_parameters is None:
        launcher_parameters = {}

    if main_parameters is None:
        main_parameters = {}

    launcher_parameters = copy.deepcopy(launcher_parameters)
    for k, v in launcher_parameters.items():
        if isinstance(v, list) and len(v) == 1:
            launcher_parameters[k] = v[0]
    manually_specified_candidates = {
        k: v for k, v in launcher_parameters.items() if isinstance(v, list)
    }
    manually_specified_parameters = {
        k: v for k, v in launcher_parameters.items() if not isinstance(v, list)
    }

    cpu_info = CPUinfo()

    if kwargs.get("convert_csv", False):
        if "logfile" not in kwargs:
            raise ValueError(
                "a log file needs to be specified when convert_csv is True"
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

    experiment = create_experiment(
        conn,
        experiment_info,
        specified_candidates=manually_specified_candidates,
        specified_parameters=manually_specified_parameters,
    )
    experiment_id = hexlify(getrandbits(32).to_bytes(4, "big")).decode("ascii")

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments_dict = dict(suggestion.assignments)

        # Setting values that were manually specified (as they will not be suggested by Sigopt)
        assignments_dict.update(manually_specified_parameters)

        # Setting the number of cores and instances if this was not set before.
        if "nb_cores" not in assignments_dict:
            assignments_dict["nb_cores"] = generate_nb_cores_candidates(
                main_parameters["batch_size"],
                mode,
                cpu_info,
                nb_instances=int(assignments_dict["nb_cores"]),
            )

        if "instances" not in assignments_dict:
            default_nb_instances = generate_nb_instances_candidates(
                main_parameters["batch_size"],
                mode,
                cpu_info,
                nb_cores=int(assignments_dict["nb_cores"]),
            )
            nb_instances = launcher_parameters.get("instances", default_nb_instances)
            assignments_dict["instances"] = nb_instances

        exp_result = launch_and_wait(
            launcher_parameters=assignments_dict,
            main_parameters=main_parameters,
            experiment_id=experiment_id,
        )
        values = {
            TuningMode.LATENCY: [{"name": "latency", "value": exp_result.latency}],
            TuningMode.THROUGHPUT: [
                {
                    "name": "throughput",
                    "value": exp_result.throughput,
                }
            ],
            TuningMode.BOTH: [
                {"name": "latency", "value": exp_result.latency},
                {"name": "throughput", "value": exp_result.throughput},
            ],
        }

        if (
            values[TuningMode.LATENCY][0]["value"] is None
            and values[TuningMode.THROUGHPUT][0]["value"] is None
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

        print("-" * 40)
        print("\nSigopt suggestion:")
        for name, assignment in suggestion.assignments.items():
            print(f"\t- {name} = {assignment}")
        print(
            f"Experiment result:\n\t- Latency = {exp_result.latency} ms\n\t- Throughput = {exp_result.throughput} it/s\n"
        )
        print("-" * 40)

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

    print("Sigopt experiment report")
    for key, value in report.items():
        print(f"\t- {key} -> {value}")

    filename = kwargs["logfile"]
    with open(filename, "a") as f:
        print(f"Saved the Sigopt experiment report at {filename}")
        json.dump(report, f)
        f.write("\n")
