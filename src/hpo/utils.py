import re
import os
import sys
import subprocess

from enum import Enum
from typing import Any, Dict, NamedTuple
from binascii import hexlify
from random import getrandbits

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
    experiment_id = hexlify(getrandbits(32).to_bytes(4, 'big')).decode('ascii')
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
    missing = []
    for arg in MANDATORY_ARGUMENTS:
        if arg not in kwargs:
            missing.append(arg)

    if missing:
        is_ = "is" if len(missing) == 1 else "are"
        missing = " ,".join(missing)
        raise ValueError(f"Tune function needs {MANDATORY_ARGUMENTS} to be specified, but {missing} {is_} missing")
