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

import pandas as pd

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
    conn,
    experiment_info,
    specified_candidates=None,
    specified_parameters=None,
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
        if param == "instances":
            continue
        candidates = list(map(str, specified_candidates[param]))
        experiment_meta["parameters"][param] = {
            "categorical_values": list(map(str, candidates)),
            "name": param,
            "type": "categorical",
        }

    idx2nb_instances = None
    idx2nb_cores = None
    if "instances" in specified_candidates:
        idx2nb_instances = specified_candidates["instances"]
        experiment_meta["parameters"]["instances"] = {
            "bounds": {"min": 1, "max": len(idx2nb_instances)},
            "name": "instances",
            "type": "int",
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
            idx2nb_cores = nb_cores
            experiment_meta["parameters"]["nb_cores"] = {
                "bounds": {"min": 1, "max": len(idx2nb_cores)},
                "name": "nb_cores",
                "type": "int",
            }

    print(experiment_meta["parameters"])
    experiment_meta["parameters"] = list(experiment_meta["parameters"].values())

    experiment = conn.experiments().create(**experiment_meta)
    print(f"Created experiment: https://app.sigopt.com/experiment/{experiment.id}")

    return experiment, idx2nb_instances, idx2nb_cores


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

    # if kwargs.get("convert_csv", False):
    #     if "logfile" not in kwargs:
    #         raise ValueError(
    #             "a log file needs to be specified when convert_csv is True"
    #         )
    #     convert_fmt(kwargs["logfile"])
    #     return

    conn = Connection()
    if "proxy" in kwargs:
        conn.set_proxies({"http": kwargs["proxy"]})

    mode = kwargs["mode"]
    exp_name = kwargs["exp_name"]

    # TODO: is it better to have everything explicitly set or just to pass kwargs as
    # experiment_info?
    experiment_info = {
        "name": exp_name,
        "mode": mode,
        "n_trials": kwargs["n_trials"],
        "batch_size": main_parameters["batch_size"],
        "sequence_length": main_parameters["sequence_length"],
    }

    project_name = kwargs.get("project", None)
    if project_name:
        experiment_info["project"] = project_name

    experiment, idx2nb_instances, idx2nb_cores = create_experiment(
        conn,
        experiment_info,
        specified_candidates=manually_specified_candidates,
        specified_parameters=manually_specified_parameters,
    )
    experiment_id = hexlify(getrandbits(32).to_bytes(4, "big")).decode("ascii")

    for _ in range(experiment.observation_budget):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments_dict = dict(suggestion.assignments)

        # Setting values that were manually specified (as they will not be suggested by Sigopt)
        assignments_dict.update(manually_specified_parameters)

        if idx2nb_instances is not None:
            assignments_dict["instances"] = idx2nb_instances[
                assignments_dict["instances"] - 1
            ]
        if idx2nb_cores is not None:
            assignments_dict["nb_cores"] = idx2nb_cores[
                assignments_dict["nb_cores"] - 1
            ]

        # Setting the number of cores and instances if this was not set before.
        if "nb_cores" not in assignments_dict:
            _, nb_cores = generate_nb_cores_candidates(
                main_parameters["batch_size"],
                mode,
                cpu_info,
                nb_instances=int(assignments_dict["instances"]),
            )

            # With Optuna, a value is selected between 1 and total_num_cores // nb_instances but
            # here total_num_cores // nb_instances is used as making a choice is no longer possible.
            assignments_dict["nb_cores"] = nb_cores[-1]

            print(
                f"Setting the number of cores to {assignments_dict['nb_cores']} for the experiment from the tuned"
                f"number of instances (Sigopt assignment is {assignments_dict['instances']})"
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

            print(
                f"Setting the number of instances to {assignments_dict['instances']} for the experiment from the tuned"
                f"number of cores (Sigopt assignment is {assignments_dict['nb_cores']})"
            )

        exp_result = launch_and_wait(
            launcher_parameters=assignments_dict,
            main_parameters=main_parameters,
            experiment_id=experiment_id,
        )

        # Dummy experiment to make testing faster.
        # from .utils import ExperimentResult
        # exp_result = ExperimentResult(latency=1.2, throughput=1.1)

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
            observation = (
                conn.experiments(experiment.id)
                .observations()
                .create(
                    suggestion=suggestion.id,
                    values=values[mode],
                )
            )
            updated_assignments = {
                k: v for k, v in assignments_dict.items() if k in suggestion.assignments
            }
            if idx2nb_instances is not None or idx2nb_cores is not None:
                conn.experiments(experiment.id).observations(observation.id).update(
                    suggestion=None, assignments=updated_assignments
                )

        experiment = conn.experiments(experiment.id).fetch()

        print("-" * 40)
        print("\nSigopt suggestion:")
        for name, assignment in updated_assignments.items():
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

    report_path = os.path.join("outputs", f"{exp_name}_sigopt_report.json")
    with open(report_path, "w") as f:
        print(f"Saved the Sigopt experiment report at {report_path}")
        json.dump(report, f)
        f.write("\n")

    report_path = os.path.join("outputs", f"{exp_name}_sigopt_report.csv")
    df = pd.DataFrame.from_dict(report)
    df.to_csv(report_path)
    print(f"And as a CSV file at {report_path}")
