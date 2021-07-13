import copy
import functools
import json
import os

import joblib
import optuna
import pandas as pd
from optuna import Trial, create_study
from optuna.importance import get_param_importances
from optuna.samplers import NSGAIISampler, TPESampler

from utils import CPUinfo

from .utils import (
    ExperimentResult,
    TuningMode,
    generate_nb_cores_candidates,
    generate_nb_instances_candidates,
    launch_and_wait,
    tune_function,
)


def _optimize_latency_and_throughput(
    mode,
    trial,
    launcher_parameters=None,
    main_parameters=None,
) -> ExperimentResult:
    """
    Main function that suggests unspecfied mandatory launcher_parameters via Optuna according to a
    TuningMode, runs an experiment using those parameters and returns the result as an
    ExperimentResult. If a launcher parameter was specified as a list, it is assumed to be a set of
    possible candidates from which Optuna should choose.
    """
    if launcher_parameters is None:
        launcher_parameters = {}

    if main_parameters is None:
        main_parameters = {}

    cpu_info = CPUinfo()

    # It is necessary to check for presence of key in launcher_parameters before actually setting
    # the default value because suggest_categorical does not support dynamic value space (which
    # could happen when setting a default value and overwriting it with the specified one)
    parameters = {
        "instances": 1,
    }
    suggested_instances = False
    if "instances" not in launcher_parameters:
        parameters["instances"] = generate_nb_instances_candidates(
            main_parameters["batch_size"], mode, cpu_info
        )
        if len(parameters["instances"]) > 1:
            parameters["instances"] = trial.suggest_categorical(
                "instances", parameters["instances"]
            )
            suggested_instances = True
        else:
            parameters["instances"] = parameters["instances"][0]

    if "openmp" not in launcher_parameters:
        parameters["openmp"] = trial.suggest_categorical("openmp", ["openmp", "iomp"])

    if "allocator" not in launcher_parameters:
        parameters["allocator"] = trial.suggest_categorical(
            "allocator", ["default", "tcmalloc", "jemalloc"]
        )

    if "huge_pages" not in launcher_parameters:
        parameters["huge_pages"] = trial.suggest_categorical(
            "huge_pages", ["on", "off"]
        )

    suggested_parameters = parameters.keys() - (
        {"instances"} if not suggested_instances else {}
    )

    def prepare_parameter_for_optuna(trial, key, value):
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            suggested_parameters.add(key)
            return trial.suggest_categorical(key, value)
        return value

    launcher_parameters = {
        k: prepare_parameter_for_optuna(trial, k, v)
        for k, v in launcher_parameters.items()
    }
    parameters.update(launcher_parameters)

    if parameters["instances"] > cpu_info.physical_core_nums:
        parameters["instances"] = cpu_info.physical_core_nums

    # Using dummy value for batch size (1) as it will not be used since the number of instances
    # is provided.
    candidates_type, nb_cores = generate_nb_cores_candidates(
        1, mode, cpu_info, nb_instances=parameters["instances"]
    )
    if candidates_type == "range":
        parameters["nb_cores"] = trial.suggest_int("nb_cores", *nb_cores)
        suggested_parameters.add("nb_cores")
    if candidates_type == "discrete":
        if len(nb_cores) > 1:
            parameters["nb_cores"] = trial.suggest_categorical("nb_cores", nb_cores)
            suggested_parameters.add("nb_cores")
        elif nb_cores:
            parameters["nb_cores"] = nb_cores[0]
        else:
            parameters["nb_cores"] = (
                parameters["instances"] // cpu_info.physical_core_nums
            )

    batch_size = main_parameters["batch_size"]
    # TODO: well define the behaviour for batch size (can we provide multiple batch sizes?)
    if isinstance(batch_size, list) and len(batch_size) > 1:
        filter_fn = lambda size: size % parameters["instances"] == 0
        batch_size = list(filter(filter_fn, batch_size))
    main_parameters["batch_size"] = batch_size

    # print("launcher_parameters", parameters)
    # print("main_parameters", main_parameters)

    exp_result = launch_and_wait(parameters, main_parameters)

    print("-" * 40)
    print("\nOptuna suggestion:")
    for name in suggested_parameters:
        print(f"\t- {name} = {parameters[name]}")
    print(
        f"Experiment result:\n\t- Latency = {exp_result.latency} ms\n\t- Throughput = {exp_result.throughput} it/s\n"
    )
    print("-" * 40)

    return exp_result


def optimize_latency_and_throughput(
    trial: Trial, launcher_parameters=None, main_parameters=None
):
    experiment_result = _optimize_latency_and_throughput(
        TuningMode.BOTH,
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
    )

    return experiment_result.latency, experiment_result.throughput


def optimize_latency(
    trial: Trial, launcher_parameters=None, main_parameters=None
) -> float:
    return _optimize_latency_and_throughput(
        TuningMode.LATENCY,
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
    ).latency


def optimize_throughput(
    trial: Trial, launcher_parameters=None, main_parameters=None
) -> float:
    return _optimize_latency_and_throughput(
        TuningMode.THROUGHPUT,
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
    ).throughput


directions = {
    TuningMode.LATENCY: {"direction": "minimize"},
    TuningMode.THROUGHPUT: {"direction": "maximize"},
    TuningMode.BOTH: {"directions": ["minimize", "maximize"]},
}


samplers = {
    TuningMode.LATENCY: TPESampler,
    TuningMode.THROUGHPUT: TPESampler,
    TuningMode.BOTH: NSGAIISampler,
}


objectives = {
    TuningMode.LATENCY: optimize_latency,
    TuningMode.THROUGHPUT: optimize_throughput,
    TuningMode.BOTH: optimize_latency_and_throughput,
}


# create_study_functions = {
#     TuningMode.LATENCY: create_study,
#     TuningMode.THROUGHPUT: create_study,
#     TuningMode.BOTH: optuna.multi_objective.create_study,
# }


@tune_function
def optuna_tune(launcher_parameters=None, main_parameters=None, **kwargs):
    mode = kwargs["mode"]
    exp_name = kwargs["exp_name"]
    n_trials = kwargs["n_trials"]

    # study_fn = create_study_functions[mode]
    sampler_cls = samplers[mode]
    direction = directions[mode]
    objective = objectives[mode]

    # study = study_fn(sampler=sampler_cls(), **direction)
    study = create_study(sampler=sampler_cls(), **direction)
    objective = functools.partial(
        objective,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    study_path = os.path.join("outputs", f"{exp_name}_study.pkl")
    joblib.dump(study, study_path)

    print(
        f"Saved study at {study_path}, this can be useful to access extra information"
    )

    report = {}
    report["Batch size"] = main_parameters["batch_size"]
    report["Sequence length"] = main_parameters["sequence_length"]
    report["Mode"] = kwargs["mode"].value
    report["Framework"] = main_parameters["backend"]
    report_path = os.path.join("outputs", f"{exp_name}_report.json")

    def multi_objective_target(trial, idx_to_target):
        return trial.values[idx_to_target]

    if mode is TuningMode.BOTH:
        try:
            param_importances_for_latency = get_param_importances(
                study, target=functools.partial(multi_objective_target, idx_to_target=0)
            )
            param_importances_for_throughput = get_param_importances(
                study, target=functools.partial(multi_objective_target, idx_to_target=1)
            )
        except ZeroDivisionError:
            print(
                "Parameter importances evaluation failed (most likely due to a too small number of trials)"
            )
            param_importances_for_latency = param_importances_for_throughput = "NA"

        importances = {
            "Latency": param_importances_for_latency,
            "Throughput": param_importances_for_throughput,
        }

        print("To get the best parameters, check the pareto front.")

        report["Parameter importances"] = importances
        report["Best trials"] = []
        for trial in study.best_trials:
            trial_result = {
                "Latency": trial.values[0],
                "Throughput": trial.values[1],
                **trial.params,
            }
            report["Best trials"].append(trial_result)

        # importances_path = os.path.join("outputs", f"{exp_name}_importances.csv")
        # df = pd.DataFrame.from_dict(importances)
        # df.to_csv(importances_path)

        pareto_front_path = os.path.join("outputs", f"{exp_name}_pareto_front.png")
        report["Pareto front path"] = pareto_front_path

        fig = optuna.visualization.plot_pareto_front(
            study, target_names=["Latency", "Throughput"]
        )
        fig.write_image(pareto_front_path)

        print(f"Saved Pareto front figure at {pareto_front_path}\n")

        print("Parameter importance for latency:")
        for param, importance in importances["Latency"].items():
            print(f"\t- {param} -> {importance}")

        print("Parameter importance for throughput:")
        for param, importance in importances["Throughput"].items():
            print(f"\t- {param} -> {importance}")

    else:
        importances = get_param_importances(study)
        report["Parameter importances"] = importances
        report["Best parameters"] = study.best_params

        print(
            "Best {}: {} (params: {})\n".format(
                mode.value.lower(), study.best_value, study.best_params
            )
        )
        print("Parameter importance:")
        for param, importance in importances.items():
            print(f"\t- {param} -> {importance}")

        # study_result = {}
        # for param in study.best_params:
        #     importance = importances.get(param, 0)
        #     param_value = study.best_params[param]
        #     study_result[param] = {"importance": importance, "value": param_value}

        # df = pd.DataFrame.from_dict(study_result)
        # df.to_csv(path)

    with open(report_path, "w") as f:
        print(f"Saved the Optuna experiment report at {report_path}")
        json.dump(report, f)
