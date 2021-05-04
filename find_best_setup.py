import os
import re
import subprocess
import sys
from enum import Enum
from typing import Any, Dict, NamedTuple

from argparse import ArgumentParser
from optuna import create_study, Trial
from optuna.samplers import TPESampler

from utils.cpu import CPUinfo

RE_LATENCY = re.compile(r'Latency: (.*)ms')
RE_THROUGHPUT = re.compile(r'Throughput: (.*)it/s')


class TuningMode(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"


ExperimentResult = NamedTuple("ExperimentResult", [("latency", float), ("throughput", float)])


def optimize_latency(trial: Trial) -> float:
    cpu_info = CPUinfo()

    parameters = {
        "instances": 1,
        "nb_cores": trial.suggest_int("nb_cores", low=1, high=cpu_info.physical_core_nums),
        "numactl": trial.suggest_categorical("numactl", ["off"]),
        "openmp": trial.suggest_categorical("openmp", ["openmp", "iomp"]),
        "allocator": trial.suggest_categorical("allocator", ["default", "tcmalloc"]),
        "huge_pages": trial.suggest_categorical("huge_pages", ["on", "off"]),
    }
    return launch_and_wait(parameters).latency


def launch_and_wait(parameters: Dict[str, Any]) -> ExperimentResult:
    cmd = [sys.executable, "launcher.py"]

    for name, value in parameters.items():
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

    # Main script parameter
    cmd += [
        "--",
        "src/main.py",
        f"batch_size={args.batch_size}",
        f"sequence_length={args.sequence_length}",
        f"backend={args.framework}"
    ]

    process = subprocess.Popen(cmd, cwd=os.curdir, stdout=subprocess.PIPE)
    process.wait()
    output = process.stdout.read().decode("utf-8")

    # Extract metrics
    latency_match = RE_LATENCY.search(output)
    throughput_match = re.search(RE_THROUGHPUT, output)

    latency = float("+inf") if latency_match is None else float(latency_match.group(1))
    throughput = float("-inf") if throughput_match is None else float(throughput_match.group(1))
    return ExperimentResult(latency, throughput)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="The number of elements in the batch")
    parser.add_argument("--sequence_length", type=int, default=128, help="The length of the sequence to evaluate")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow", "torchscript", "xla", "ort"], help="The framework to evaluate")
    parser.add_argument("--mode", type=TuningMode, choices=[TuningMode.LATENCY, TuningMode.THROUGHPUT], help="The criteria to use for the benchmark")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of experiments to run")

    # Parser arguments
    args = parser.parse_args()

    # Create some dynamic args
    args.study = create_study(
        sampler=TPESampler(),
        direction="minimize" if args.mode == TuningMode.LATENCY else "maximize"
    )
    args.study.optimize(
        optimize_latency,
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    print("Best {}: {} (params: {})\n".format(args.mode.value.lower(), args.study.best_value, args.study.best_params))
    print()


