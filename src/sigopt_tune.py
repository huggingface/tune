import os
import re
import subprocess
import sys
from enum import IntEnum
from typing import Any, Dict, NamedTuple
from argparse import ArgumentParser
from utils.cpu import CPUinfo
from sigopt import Connection


RE_LATENCY = re.compile(r'Latency: (.*)ms')
RE_THROUGHPUT = re.compile(r'Throughput: (.*)it/s')


class TuningMode(IntEnum):
    LATENCY = 0
    THROUGHPUT = 1
    BOTH = 2


ExperimentResult = NamedTuple("ExperimentResult", [("latency", float), ("throughput", float)])


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
        "main.py",
        "benchmark_duration=60",
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

    latency = None if latency_match is None else float(latency_match.group(1))
    throughput = None if throughput_match is None else float(throughput_match.group(1))
    # return ExperimentResult(latency, throughput)
    return [{'name': 'latency', 'value': latency}, {'name': 'throughput', 'value': throughput}]


def create_exp(conn, cfg):
    metrics = [{'name': 'latency', 'objective': 'minimize'},
               {'name': 'throughput', 'objective': 'maximize'}]
    cpu_info = CPUinfo()

    experiment_meta = {
        'name': 'hf-tune',
        'project': 'huggingface',
        'parameters': [
            {'bounds': {'min': 1, 'max': cpu_info.physical_core_nums}, 'name': 'nb_cores', 'type': 'int'},
            {'categorical_values': ['openmp', 'iomp'], 'name': 'openmp', 'type': 'categorical'},
            {'categorical_values': ['default', 'tcmalloc'], 'name': 'allocator', 'type': 'categorical'},
            {'categorical_values': ['on', 'off'], 'name': 'huge_pages', 'type': 'categorical'},

        ],
        'observation_budget': cfg.n_trials,
        'metrics': [metrics[cfg.mode]] if cfg.mode < TuningMode.BOTH else metrics,
        'parallel_bandwidth': 1,
    }

    if cfg.mode != TuningMode.LATENCY:
        instances = [1]
        while cfg.batch_size / instances[-1] > 1 and cpu_info.physical_core_nums / instances[-1] > 1:
            instances.append(instances[-1] * 2)
        instances = [str(i) for i in instances]
        if len(instances) > 1:
            experiment_meta['parameters'].append(
                {'categorical_values': instances, 'name': 'instances', 'type': 'categorical'}
            )

    experiment = conn.experiments().create(**experiment_meta)
    print("created experiment: https://app.sigopt.com/experiment/" + experiment.id)
    return experiment


def sigopt_tune(cfg):
    conn = Connection(client_token="WECSOKFVSWZNKNOPOMMIILBOXJHNMOBWPIKLKEGXSDRPCIGR")
    conn.set_proxies({'http': cfg.proxy}) if cfg.proxy is not None else None
    experiment = create_exp(conn, cfg)
    numactl_disable = {True: 'off', False: 'on'}

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        suggestion.assignments['numactl'] = numactl_disable[cfg.disable_numactl]
        if hasattr(suggestion.assignments, 'instances'):
            suggestion.assignments['instances'] = int(suggestion.assignments['instances'])
        else:
            suggestion.assignments['instances'] = 1
        values = launch_and_wait(suggestion.assignments)
        print('suggestion: %s, values: %s' % (suggestion.assignments, values))
        if values[0]['value'] is None and values[1]['value'] is None:
            conn.experiments(experiment.id).suggestions(suggestion.id).delete()
            print('incomplete metrics for suggestion: %s' % suggestion.assignments)
        else:
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                values=[values[cfg.mode]] if cfg.mode != TuningMode.BOTH else values,
            )
        experiment = conn.experiments(experiment.id).fetch()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="The number of elements in the batch")
    parser.add_argument("--sequence_length", type=int, default=128, help="The length of the sequence to evaluate")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow", "torchscript", "xla", "ort"],
                        required=True, help="The framework to evaluate")
    parser.add_argument("--mode", type=int, choices=[TuningMode.LATENCY, TuningMode.THROUGHPUT, TuningMode.BOTH],
                        required=True, help="The criteria to use for the benchmark")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of experiments to run")
    parser.add_argument("--disable_numactl", action="store_true", help="Disable the usage of numactl")
    parser.add_argument("--proxy", type=str, default=None, help="Proxies for sigopt")

    # Parser arguments
    args = parser.parse_args()
    sigopt_tune(args)
    print('done!')
