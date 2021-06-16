import os
import re
import subprocess
import sys
import functools
import pandas as pd
from enum import Enum
from typing import Any, Dict, NamedTuple

from argparse import ArgumentParser
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


def _optimize_latency_and_throughput(trial: Trial, launcher_parameters=None, main_parameters=None, optimize_throughput=True):
    if launcher_parameters is None:
        launcher_parameters = {}

    if main_parameters is None:
        main_parameters = {}

    cpu_info = CPUinfo()
    instances = _compute_instance(4, optimize_throughput, cpu_info) # TODO: how to add batch size.
    if isinstance(instances, list):
        instances = trial.suggest_categorical("instances", instances)

    parameters = {
        "instances": instances,
        "nb_cores": trial.suggest_int(
            "nb_cores", low=1, high=cpu_info.physical_core_nums
        ),
        "openmp": trial.suggest_categorical("openmp", ["openmp", "iomp"]),
        "allocator": trial.suggest_categorical("allocator", ["default", "tcmalloc"]),
        "huge_pages": trial.suggest_categorical("huge_pages", ["on", "off"]),
    }

    parameters.update(**launcher_parameters)

    return launch_and_wait(parameters, main_parameters)


def optimize_latency_and_throughput(trial: Trial, launcher_parameters=None, main_parameters=None):
    experiment_result = _optimize_latency_and_throughput(
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=True,
    )
    return experiment_result.latency, experiment_result.throughput



def optimize_latency(trial: Trial, launcher_parameters=None, main_parameters=None) -> float:
    return _optimize_latency_and_throughput(
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=False
    ).latency


def optimize_throughput(trial: Trial, launcher_parameters=None, main_parameters=None) -> float:
    return _optimize_latency_and_throughput(
        trial,
        launcher_parameters=launcher_parameters,
        main_parameters=main_parameters,
        optimize_throughput=True
    ).throughput


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
    }

    prepared_main_parameters.update(main_parameters)

    def prepare_value(value):
        if isinstance(value, (list, tuple)):
            return ",".join(map(str, value))
        return value

    prepared_main_parameters = [
        f"{k}={prepare_value(v)}" for k, v in prepared_main_parameters.items()
    ]

    multirun = "--multirun" if any("," in v for v in prepared_main_parameters) else ""
    # Main script parameter
    cmd += ["--", "src/main.py", multirun] + prepared_main_parameters

    print(f"Running command: {''.join(cmd)}")

    process = subprocess.Popen(cmd, cwd=os.curdir, stdout=subprocess.PIPE)
    process.wait()
    output = process.stdout.read().decode("utf-8")

    # Extract metrics
    latency_match = RE_LATENCY.search(output)
    throughput_match = re.search(RE_THROUGHPUT, output)

    latency = float("+inf") if latency_match is None else float(latency_match.group(1))
    throughput = (
        float("-inf") if throughput_match is None else float(throughput_match.group(1))
    )
    return ExperimentResult(latency, throughput)


def hpo(
    mode: TuningMode,
    n_trials: int = 15,
    launcher_parameters=None,
    main_parameters=None,
    exp_name=None,
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

    study = create_study(
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


    def multi_objective_target(trial):
        # TODO: find the proper way to implement target.
        return trial.values[0]

    mode2target = {
        TuningMode.LATENCY: None,
        TuningMode.THROUGHPUT: None,
        TuningMode.LATENCY_AND_THROUGHPUT: multi_objective_target
    }

    importances = get_param_importances(study, target=mode2target[mode])

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

    filename = f"{exp_name}_optuna_results.csv" if exp_name else "optuna_results.csv"
    path = os.path.join("outputs", filename)
    df = pd.DataFrame.from_dict(study_result)
    df.to_csv(path)


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
        choices=[TuningMode.LATENCY, TuningMode.THROUGHPUT, TuningMode.LATENCY_AND_THROUGHPUT],
        help="The criteria to use for the benchmark",
    )
    parser.add_argument(
        "--n_trials", type=int, default=15, help="Number of experiments to run"
    )
    parser.add_argument(
        "--disable_numactl", action="store_true", help="Disable the usage of numactl"
    )

    parser.add_argument(
        "--exp_name", type=str, default=None, help="The name of the experiment"
    )

    # Parser arguments
    args = parser.parse_args()

    main_parameters = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "backend": args.framework,
    }
    hpo(
        args.mode,
        args.n_trials,
        launcher_parameters={"numactl": "off" if args.disable_numactl else "on"},
        main_parameters=main_parameters,
        exp_name=args.exp_name,
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
