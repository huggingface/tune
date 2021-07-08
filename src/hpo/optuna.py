import copy
import functools
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


def prepare_parameter_for_optuna(trial, key, value):
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return trial.suggest_categorical(key, value)
    return value


def specialize_objective(optimize_fn, launcher_parameters=None, main_parameters=None):
    """
    Specializes an objective function (optimize_latency, optimize_throughput,
    optimize_latency_and_throughput) by partially applying main_parameters and launcher_parameters
    and returning the resulting objective function.
    """

    if launcher_parameters is None:
        launcher_parameters = {}

    launcher_parameters = copy.deepcopy(launcher_parameters)
    main_parameters = copy.deepcopy(main_parameters)

    @functools.wraps(optimize_fn)
    def wrapper(trial: Trial):
        prepared_launcher_parameters = {
            k: prepare_parameter_for_optuna(trial, k, v)
            for k, v in launcher_parameters.items()
        }
        return optimize_fn(
            trial,
            launcher_parameters=prepared_launcher_parameters,
            main_parameters=main_parameters,
        )

    return wrapper


def _optimize_latency_and_throughput(
    mode,
    trial,
    launcher_parameters=None,
    main_parameters=None,
) -> ExperimentResult:
    """
    Main function that suggests unspecfied mandatory launcher_parameters via Optuna according to a
    TuningMode, runs an experiment using those parameters and returns the result as an
    ExperimentResult.
    """
    if launcher_parameters is None:
        launcher_parameters = {}

    if main_parameters is None:
        # TODO: add safety check to make sure all the mandatory keys are provided.
        main_parameters = {}

    cpu_info = CPUinfo()

    parameters = {
        "instances": 1,
    }

    # It is necessary to check for presence of key in launcher_parameters before actually setting
    # the default value because suggest_categorical does not support dynamic value space (which
    # could happen when setting a default value and overwriting it with the specified one)

    if "instances" not in launcher_parameters:
        parameters["instances"] = generate_nb_instances_candidates(
            main_parameters["batch_size"], mode, cpu_info
        )
        if len(parameters["instances"]) > 1:
            parameters["instances"] = trial.suggest_categorical(
                "instances", parameters["instances"]
            )
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
    if candidates_type == "discrete":
        if len(nb_cores) > 1:
            parameters["nb_cores"] = trial.suggest_categorical("nb_cores", nb_cores)
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

    print("launcher_parameters", parameters)
    print("main_parameters", main_parameters)

    return launch_and_wait(parameters, main_parameters)


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


create_study_functions = {
    TuningMode.LATENCY: create_study,
    TuningMode.THROUGHPUT: create_study,
    TuningMode.BOTH: optuna.multi_objective.create_study,
}


@tune_function
def optuna_tune(launcher_parameters=None, main_parameters=None, **kwargs):
    mode = kwargs["mode"]
    exp_name = kwargs["exp_name"]
    n_trials = kwargs["n_trials"]

    study_fn = create_study_functions[mode]
    sampler_cls = samplers[mode]
    direction = directions[mode]
    objective = objectives[mode]

    study = study_fn(sampler=sampler_cls(), **direction)
    objective = specialize_objective(
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

    def multi_objective_target(trial, idx_to_target):
        return trial.values[idx_to_target]

    if mode is TuningMode.BOTH:
        param_importances_for_latency = get_param_importances(
            study, target=functools.partial(multi_objective_target, idx_to_target=0)
        )
        param_importances_for_throughput = get_param_importances(
            study, target=functools.partial(multi_objective_target, idx_to_target=1)
        )
        importances = {
            "Latency": param_importances_for_latency,
            "Throughput": param_importances_for_throughput,
        }

        print("To get the best parameters, check the pareto front")

        print("Parameter importance for latency:")
        for param, importance in importances["Latency"].items():
            print(f"\t- {param} -> {importance}")

        print("Parameter importance for throughput:")
        for param, importance in importances["Throughput"].items():
            print(f"\t- {param} -> {importance}")

        importances_path = os.path.join("outputs", f"{exp_name}_importances.csv")
        df = pd.DataFrame.from_dict(importances)
        df.to_csv(importances_path)

        pareto_front_path = os.path.join("outputs", f"{exp_name}_pareto_front.png")
        fig = optuna.multi_objective.visualization.plot_pareto_front(
            study, names=["Latency", "Throughput"]
        )
        fig.write_image(pareto_front_path)

        print(
            f"Saved parameter importances at {importances_path}"
            f" and Pareto front figure at {pareto_front_path}"
        )

    else:
        importances = get_param_importances(study)
        print(
            "Best {}: {} (params: {})\n".format(
                mode.value.lower(), study.best_value, study.best_params
            )
        )
        print("Parameter importance:")
        for param, importance in importances.items():
            print(f"\t- {param} -> {importance}")

        study_result = {}
        for param in study.best_params:
            importance = importances.get(param, 0)
            param_value = study.best_params[param]
            study_result[param] = {"importance": importance, "value": param_value}

        filename = f"{exp_name}_importances_and_values.csv"
        path = os.path.join("outputs", filename)

        df = pd.DataFrame.from_dict(study_result)
        df.to_csv(path)

        print(f"Saved parameter importances and values at {path}")
