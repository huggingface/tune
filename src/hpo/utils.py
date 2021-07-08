import copy
import os
import re
import subprocess
import sys
from binascii import hexlify
from enum import Enum
from functools import wraps
from random import getrandbits
from typing import Any, Dict, List, Tuple, NamedTuple

RE_LATENCY = re.compile(r"Latency: (.*)ms")
RE_THROUGHPUT = re.compile(r"Throughput: (.*)it/s")

MANDATORY_ARGUMENTS = {"exp_name", "n_trials", "mode"}


class TuningMode(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BOTH = "both"


ExperimentResult = NamedTuple(
    "ExperimentResult", [("latency", float), ("throughput", float)]
)


def aggregate_latencies(latencies):
    """
    Aggregates latencies across multiple instances.
    The aggregation rule for latencies is taking the max among them.
    """
    if not latencies:
        return float("+inf")
    return max(map(float, latencies))


def aggregate_throughputs(throughputs):
    """
    Aggregates throughputs across multiple instances.
    The aggregation rule for throughputs is summing them together.
    """
    if not throughputs:
        return float("-inf")
    return sum(map(float, throughputs))


def generate_nb_instances_candidates(
    batch_size, mode, cpu_info, nb_cores=-1
) -> List[int]:
    """
    Returns a list of candidates for the number of instances.
    If the number of cores is known(nb_cores > 0), then the good candidates is
    total_num_cores // nb_cores where total_num_cores = cpu_info.physical_core_nums, else a good
    candidate is one that divides batch_size.
    """
    total_num_cores = cpu_info.physical_core_nums
    if nb_cores > 0:
        return total_num_cores // nb_cores
    instances = [1]
    if mode is not TuningMode.LATENCY:
        for i in range(2, total_num_cores + 1):
            if batch_size % i != 0:
                continue
            instances.append(i)
    return instances


def generate_nb_cores_candidates(
    batch_size, mode, cpu_info, nb_instances=-1
) -> Tuple[str, List[int]]:
    """
    Returns a list of candidates for the number of cores (per instance).
    If the number of instances is known (nb_instances > 0), then a good candidate is one in
    [1, total_num_cores // nb_instances + 1] where total_num_cores = cpu_info.physical_core_nums,
    else the assumption made is that the number of instances will be
    total_num_cores // selected_candidate, so in that case a candidate must satisfy:
        1. candidate divides total_num_cores
        2. total_num_cores // candidate divides batch_size
    """
    total_num_cores = cpu_info.physical_core_nums
    if nb_instances > 0:
        return "range", [1, total_num_cores // nb_instances]
    elif mode is TuningMode.LATENCY:
        return "range", [1, total_num_cores]
    else:
        num_cores = []
        for i in range(1, total_num_cores + 1):
            if total_num_cores % i == 0 and batch_size % (total_num_cores // i) == 0:
                num_cores.append(i)
        return "discrete", num_cores


def launch_and_wait(
    launcher_parameters: Dict[str, Any],
    main_parameters: Dict[str, Any],
    experiment_id: str = None,
) -> ExperimentResult:
    """
    Main utility function that parses launcher_parameters and main_parameters into a command,
    runs it, and returns the latency and throughput as an ExperimentResult.
    """
    if experiment_id is None:
        experiment_id = hexlify(getrandbits(32).to_bytes(4, "big")).decode("ascii")
    cmd = [sys.executable, "launcher.py", f"--experiment_id={experiment_id}"]

    for name, value in launcher_parameters.items():
        # Number of cores
        if name == "nb_cores":
            cmd.append(f"--ncore_per_instance={value}")

        # numactl
        elif name == "numactl" and value.lower() == "off":
            cmd.append("--disable_numactl")

        # Multi instances
        elif name == "instances":
            if value > 1:
                cmd += ["--multi_instance"]
            cmd += [f"--ninstances={value}"]

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

        # kmp_blocktime
        elif name == "kmp_blocktime":
            cmd.append(f"--kmp_blocktime={value}")

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


def check_tune_function_kwargs(kwargs):
    """
    Helper function that checks if a tune function (for instance optuna_tune or sigopt_tune)
    kwargs contain mandatory arguments, check MANDATORY_ARGUMENTS to know which ones.
    """

    missing = []
    for arg in MANDATORY_ARGUMENTS:
        if arg not in kwargs:
            missing.append(arg)

    if missing:
        is_ = "is" if len(missing) == 1 else "are"
        missing = " ,".join(missing)
        raise ValueError(
            f"Tune function needs {MANDATORY_ARGUMENTS} to be specified, but {missing} {is_} missing"
        )


def tune_function(func):
    """
    Decorator that wraps a function designed to be a tune function with safety checks to make sure
    proper arguments were specified.
    """

    @wraps(func)
    def wrapper(launcher_parameters=None, main_parameters=None, **kwargs):
        check_tune_function_kwargs(kwargs)
        kwargs = copy.deepcopy(kwargs)
        return func(
            launcher_parameters=launcher_parameters,
            main_parameters=main_parameters,
            **kwargs,
        )

    return wrapper
