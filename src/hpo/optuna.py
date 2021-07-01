import copy
import os
import functools
import joblib

import pandas as pd

import optuna
from optuna import create_study, Trial
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler, NSGAIISampler

from .utils import TuningMode, launch_and_wait, check_tune_function_kwargs
from utils import CPUinfo

def prepare_parameter_for_optuna(trial, key, value):
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return trial.suggest_categorical(key, value)
    return value


def specialize_objective(optimize_fn, launcher_parameters=None, main_parameters=None):

    if launcher_parameters is None:
        launcher_parameters = {}

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


def _compute_instance(batch_size, optimize_throughput, cpu_info):
    instances = 1
    if optimize_throughput:
        instances = [1]
        while (
            batch_size / instances[-1] > 1
            and cpu_info.physical_core_nums / instances[-1] > 1
        ):
            instances.append(instances[-1] * 2)
    return instances


def _optimize_latency_and_throughput(
    trial: Trial,
    launcher_parameters=None,
    main_parameters=None,
    optimize_throughput=True,
):
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
        parameters["instances"] = _compute_instance(
            main_parameters["batch_size"], optimize_throughput, cpu_info
        )
        if isinstance(parameters["instances"], list):
            parameters["instances"] = trial.suggest_categorical(
                "instances", parameters["instances"]
            )

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

    parameters["nb_cores"] = trial.suggest_int(
        "nb_cores", low=1, high=cpu_info.physical_core_nums // parameters["instances"]
    )

    batch_size = main_parameters["batch_size"]
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
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=True,
    )

    return experiment_result.latency, experiment_result.throughput


def optimize_latency(
    trial: Trial, launcher_parameters=None, main_parameters=None
) -> float:
    return _optimize_latency_and_throughput(
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=False,
    ).latency


def optimize_throughput(
    trial: Trial, launcher_parameters=None, main_parameters=None
) -> float:
    return _optimize_latency_and_throughput(
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=True,
    ).throughput


def optuna_tune(main_parameters=None, launcher_parameters=None, **kwargs):
    check_tune_function_kwargs(kwargs)
    kwargs = copy.deepcopy(kwargs)

    mode2directions = {
        TuningMode.LATENCY: {"direction": "minimize"},
        TuningMode.THROUGHPUT: {"direction": "maximize"},
        TuningMode.BOTH: {"directions": ["minimize", "maximize"]},
    }

    mode2sampler = {
        TuningMode.LATENCY: TPESampler,
        TuningMode.THROUGHPUT: TPESampler,
        TuningMode.BOTH: NSGAIISampler,
    }

    mode2create_study = {
        TuningMode.LATENCY: create_study,
        TuningMode.THROUGHPUT: create_study,
        TuningMode.BOTH: optuna.multi_objective.create_study,
    }

    study = mode2create_study[mode](
        sampler=mode2sampler[mode](),
        **mode2directions[mode],
    )

    mode2objective = {
        TuningMode.LATENCY: optimize_latency,
        TuningMode.THROUGHPUT: optimize_throughput,
        TuningMode.BOTH: optimize_latency_and_throughput,
    }

    objective = specialize_objective(
        mode2objective[mode],
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

    if mode is TuningMode.LATENCY_AND_THROUGHPUT:
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

        # TODO: print parameter importances.

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
