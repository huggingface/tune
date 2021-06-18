import os
import re
import subprocess
import sys
import functools
import joblib
import pandas as pd
from enum import Enum
from typing import Any, Dict, NamedTuple

from argparse import ArgumentParser
import optuna
from optuna import create_study, Trial
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler, NSGAIISampler

from utils.cpu import CPUinfo

RE_LATENCY = re.compile(r"Latency: (.*)ms")
RE_THROUGHPUT = re.compile(r"Throughput: (.*)it/s")


class TuningMode(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    LATENCY_AND_THROUGHPUT = "latency_and_throughput"


ExperimentResult = NamedTuple(
    "ExperimentResult", [("latency", float), ("throughput", float)]
)


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

    return instances[-1]


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


def aggregate_latencies(latencies):
    if not latencies:
        return float("+inf")
    return max(map(float, latencies))


def aggregate_throughputs(throughputs):
    if not throughputs:
        return float("-inf")
    return sum(map(float, throughputs))


def launch_and_wait(
    launcher_parameters: Dict[str, Any], main_parameters: Dict[str, Any]
) -> ExperimentResult:
    cmd = [sys.executable, "launcher.py"]

    for name, value in launcher_parameters.items():
        # Number of cores
        if name == "nb_cores":
            cmd.append(f"--ncore_per_instance={value}")

        # numactl
        elif name == "numactl" and value.lower() == "off":
            cmd.append("--disable_numactl")

        # Multi instances
        elif name == "instances" and value > 1:
            cmd += ["--multi_instance", f"--ninstances={value}"]

        # OpenMP
        elif name == "openmp" and value == "iomp":
            cmd.append("--enable_iomp")

        # Memory allocator
        elif name == "allocator":
            if value == "default":
                cmd.append("--use_default_allocator")
            else:
                cmd.append(f"--enable_{value}")

        # Transparent huge pages
        elif name == "huge_pages" and value.lower() == "on":
            cmd.append("--enable_thp")

    prepared_main_parameters = {
        "benchmark_duration": 60,
        "batch_size": 1,
        "sequence_length": 128,
        "backend": "pytorch",
    }

    prepared_main_parameters.update(main_parameters)

    def prepare_value(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return value[0]
            return ",".join(map(str, value))
        return value

    prepared_main_parameters = [
        f"{k}={prepare_value(v)}" for k, v in prepared_main_parameters.items()
    ]

    multirun = ["--multirun"] if any("," in v for v in prepared_main_parameters) else []

    # Main script parameter
    cmd += ["--", "src/main.py"] + multirun + prepared_main_parameters

    print(f"Running command: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, cwd=os.curdir, stdout=subprocess.PIPE)
    process.wait()
    output = process.stdout.read().decode("utf-8")

    # Extract metrics
    latency_matchs = re.finditer(RE_LATENCY, output)
    latencies = []
    for match in latency_matchs:
        latencies.append(match.group(1))

    latency = aggregate_latencies(latencies)

    throughput_matchs = re.finditer(RE_THROUGHPUT, output)
    throughputs = []
    for match in throughput_matchs:
        throughputs.append(match.group(1))

    throughput = aggregate_throughputs(latencies)

    return ExperimentResult(latency, throughput)


def hpo(
    exp_name,
    mode: TuningMode,
    n_trials: int = 15,
    launcher_parameters=None,
    main_parameters=None,
):

    mode2directions = {
        TuningMode.LATENCY: {"direction": "minimize"},
        TuningMode.THROUGHPUT: {"direction": "maximize"},
        TuningMode.LATENCY_AND_THROUGHPUT: {"directions": ["minimize", "maximize"]},
    }

    mode2sampler = {
        TuningMode.LATENCY: TPESampler,
        TuningMode.THROUGHPUT: TPESampler,
        TuningMode.LATENCY_AND_THROUGHPUT: NSGAIISampler,
    }

    mode2create_study = {
        TuningMode.LATENCY: create_study,
        TuningMode.THROUGHPUT: create_study,
        TuningMode.LATENCY_AND_THROUGHPUT: optuna.multi_objective.create_study,
    }

    study = mode2create_study[mode](
        sampler=mode2sampler[mode](),
        **mode2directions[mode],
    )

    mode2objective = {
        TuningMode.LATENCY: optimize_latency,
        TuningMode.THROUGHPUT: optimize_throughput,
        TuningMode.LATENCY_AND_THROUGHPUT: optimize_latency_and_throughput,
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
        importances = get_param_importances(study, target=mode2target[mode])
        # print(
        #     "Best {}: {} (params: {})\n".format(
        #         mode.value.lower(), study.best_value, study.best_params
        #     )
        # )
        # print("Parameter importance:")
        # for param, importance in importances.items():
        #     print(f"\t- {param} -> {importance}")

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The number of elements in the batch"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="The length of the sequence to evaluate",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pytorch", "tensorflow", "torchscript", "xla", "ort"],
        help="The framework to evaluate",
    )
    parser.add_argument(
        "--mode",
        type=TuningMode,
        choices=[
            TuningMode.LATENCY,
            TuningMode.THROUGHPUT,
            TuningMode.LATENCY_AND_THROUGHPUT,
        ],
        help="The criteria to use for the benchmark",
    )
    parser.add_argument(
        "--n_trials", type=int, default=15, help="Number of experiments to run"
    )
    parser.add_argument(
        "--disable_numactl", action="store_true", help="Disable the usage of numactl"
    )

    parser.add_argument(
        "--exp_name", type=str, default="default", help="The name of the experiment"
    )

    # Parser arguments
    args = parser.parse_args()

    main_parameters = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "backend": args.framework,
    }
    hpo(
        args.exp_name,
        args.mode,
        args.n_trials,
        launcher_parameters={"numactl": "off" if args.disable_numactl else "on"},
        main_parameters=main_parameters,
    )

    # # Create some dynamic args
    # args.study = create_study(
    #     sampler=TPESampler(),
    #     direction="minimize" if args.mode == TuningMode.LATENCY else "maximize"
    # )
    # args.study.optimize(
    #     optimize_latency if args.mode == TuningMode.LATENCY else optimize_throughput,
    #     n_trials=args.n_trials,
    #     show_progress_bar=True
    # )

    # importances = get_param_importances(args.study)
    # print("Best {}: {} (params: {})\n".format(args.mode.value.lower(), args.study.best_value, args.study.best_params))
    # print("Parameter importance:")
    # for param, importance in importances.items():
    #     print(f"\t- {param} -> {importance}")

    # study_result = {}
    # for param in args.study.best_params:
    #     importance = importances.get(param, 0)
    #     param_value = args.study.best_params[param]
    #     study_result[param] = {"importance": importance, "value": param_value}

    # filename = f"{args.exp_name}_optuna_results.csv" if args.exp_name else "optuna_results.csv"
    # path = os.path.join("outputs", filename)
    # df = pd.DataFrame.from_dict(study_result)
    # df.to_csv(path)
