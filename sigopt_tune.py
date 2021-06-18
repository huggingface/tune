import os
import subprocess
import sys
from enum import IntEnum
from typing import Any, Dict, NamedTuple
from argparse import ArgumentParser
from utils.cpu import CPUinfo
from sigopt import Connection
from consolidate import gather_results, aggregate_multi_instances_results, SCALING_CHOICES
from random import getrandbits
from binascii import hexlify
from glob import glob
from pathlib import Path
from functools import reduce
import json
from collections import OrderedDict


class TuningMode(IntEnum):
    LATENCY = 0
    THROUGHPUT = 1
    BOTH = 2


def convert_fmt(file):
    target = os.path.splitext(file)[0] + '.csv'
    with open(file) as f, open(target, 'w') as new_f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.rstrip())
            if i == 0:
                new_f.write(','.join(line.keys()) + '\n')
            new_f.write(','.join([str(v) for v in line.values()]) + '\n')


def launch_and_wait(parameters: Dict[str, Any]):
    args.experiment_id = hexlify(getrandbits(32).to_bytes(4, 'big')).decode('ascii')
    cmd = [sys.executable, "launcher.py", f"--experiment_id={args.experiment_id}"]

    for name, value in parameters.items():
        # Number of cores
        if name == "nb_cores":
            cmd.append(f"--ncore_per_instance={value}")

        # Multi instances
        if name == "ninstances":
            cmd += ["--multi_instance", f"--ninstances={value}"] if value > 1 else ["--ninstances=1"]

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

    # numactl
    if args.disable_numactl:
        cmd.append("--disable_numactl")

    bs = args.batch_size if args.mode != TuningMode.LATENCY else args.batch_size // parameters["ninstances"]

    # Main script parameter
    cmd += [
        "--",
        "src/main.py",
        "benchmark_duration=60",
        "warmup_runs=5",
        f"batch_size={bs}",
        f"sequence_length={args.sequence_length}",
        f"backend={args.framework}"
    ]

    print("----- launcher cmd: [%s]" % " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, cwd=os.curdir, stdout=subprocess.PIPE)
    process.wait()
    output = process.stdout.read().decode("utf-8")

    scaling_choices = ["batch-size-scaling", "core-count-scaling"]
    latency, throughput = parse_results(args.experiment_id, scaling_choices[args.mode])

    # return ExperimentResult(latency, throughput)
    return [{'name': 'latency', 'value': latency}, {'name': 'throughput', 'value': throughput}]


def parse_results(experiment_id, multi_instances_scaling=None):
    results_folder = Path("outputs/default/" + experiment_id)
    try:
        # Detect folder run type from folder structure
        instances_folder = glob(f"{results_folder.as_posix()}/*")
        is_multi_instances = len(instances_folder) > 1

        # If we detect multi instance and no scaling mode is provided, ask for a value
        if is_multi_instances and multi_instances_scaling is None:
            print("warning: multi_instances_scaling is not specified for multi-instance run.", flush=True)

        # Gather the results to manipulate
        consolidated_df, sorting_columns = gather_results(results_folder, is_multi_instances)

        if is_multi_instances and multi_instances_scaling is not None:
            agg_df = aggregate_multi_instances_results(consolidated_df, sorting_columns, multi_instances_scaling)
            latency = float(agg_df['latency_mean']['mean'].iloc[0] / 1e6)
            throughput = float(agg_df['throughput']['sum'].iloc[0])
        else:
            latency = float(consolidated_df['latency_mean'].iloc[0] / 1e6)
            throughput = float(consolidated_df['throughput'].iloc[0])

    except ValueError as ve:
        print(ve, flush=True)
        latency, throughput = None, None

    return latency, throughput


def create_exp(conn, cfg):
    metrics = [{'name': 'latency', 'objective': 'minimize'},
               {'name': 'throughput', 'objective': 'maximize'}]
    cpu_info = CPUinfo()

    def factors(n, bs, mode):
        f = sorted(list(reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))
        return [str(i) for i in f if bs * i >= n and bs % (n / i) == 0] if mode == 0 else [str(i) for i in f]

    experiment_meta = {
        'name': f'{cfg.exp_name}-mode-{cfg.mode}-bs-{cfg.batch_size}-seq-{cfg.sequence_length}',
        'project': 'huggingface',
        'parameters': [
            {'categorical_values': ['openmp', 'iomp'], 'name': 'openmp', 'type': 'categorical'},
            {'categorical_values': ['default', 'tcmalloc', 'jemalloc'], 'name': 'allocator', 'type': 'categorical'},
            {'categorical_values': ['on', 'off'], 'name': 'huge_pages', 'type': 'categorical'},
            {'categorical_values': ['0', '1'], 'name': 'kmp_blocktime', 'type': 'categorical'},
        ],
        'observation_budget': cfg.n_trials,
        'metrics': [metrics[cfg.mode]] if cfg.mode < TuningMode.BOTH else metrics,
        'parallel_bandwidth': 1,
    }

    factor_list = factors(cpu_info.physical_core_nums, cfg.batch_size, cfg.mode)
    if len(factor_list) > 1:
        experiment_meta['parameters'].append(
            {'categorical_values': factor_list, 'name': 'nb_cores', 'type': 'categorical'}
        )

    experiment = conn.experiments().create(**experiment_meta)
    print("created experiment: https://app.sigopt.com/experiment/" + experiment.id, flush=True)
    return experiment


def sigopt_tune(cfg):

    if cfg.convert_csv:
        convert_fmt(cfg.logfile)
        return

    conn = Connection()
    conn.set_proxies({'http': cfg.proxy}) if cfg.proxy is not None else None
    experiment = create_exp(conn, cfg)

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments_dict = dict(suggestion.assignments)
        if "nb_cores" not in assignments_dict.keys():
            assignments_dict["nb_cores"] = CPUinfo().physical_core_nums
        assignments_dict["ninstances"] = CPUinfo().physical_core_nums // int(assignments_dict["nb_cores"])

        values = launch_and_wait(assignments_dict)
        print('suggestion: %s, values: %s' % (suggestion.assignments, values), flush=True)
        if values[0]['value'] is None and values[1]['value'] is None:
            conn.experiments(experiment.id).suggestions(suggestion.id).delete()
            print('incomplete metrics for suggestion: %s' % suggestion.assignments, flush=True)
        else:
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                values=[values[cfg.mode]] if cfg.mode != TuningMode.BOTH else values,
            )
        experiment = conn.experiments(experiment.id).fetch()

    # dump the best result into file
    best_assignment = list(conn.experiments(experiment.id).best_assignments().fetch().iterate_pages())[0]
    report = OrderedDict(best_assignment.assignments)
    if "nb_cores" not in report.keys():
        report["nb_cores"] = CPUinfo().physical_core_nums
    report['metrics_name'] = best_assignment.values[0].name
    report['metrics_value'] = best_assignment.values[0].value
    report['batch_size'] = cfg.batch_size
    report['sequence_length'] = cfg.sequence_length
    report['mode'] = cfg.mode
    report['framework'] = cfg.framework
    report['experiment_id'] = cfg.experiment_id
    with open(cfg.logfile, 'a') as f:
        json.dump(report, f)
        f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="The number of elements in the batch")
    parser.add_argument("--sequence_length", type=int, default=128, help="The length of the sequence to evaluate")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow", "torchscript", "xla", "ort"],
                        required="--convert_csv" not in sys.argv, help="The framework to evaluate")
    parser.add_argument("--mode", type=int, choices=[TuningMode.LATENCY, TuningMode.THROUGHPUT, TuningMode.BOTH],
                        required="--convert_csv" not in sys.argv, help="The criteria to use for the benchmark")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of experiments to run")
    parser.add_argument("--disable_numactl", action="store_true", help="Disable the usage of numactl")
    parser.add_argument("--proxy", type=str, default=None, help="Proxies for sigopt")
    parser.add_argument("--exp_name", type=str, default="hf-tune", help="Experiment name for sigopt")
    parser.add_argument("--logfile", type=str, default="tune.log", help="Log file")
    parser.add_argument("--convert_csv", action="store_true", help="Convert json log file to csv")

    # Parser arguments
    args = parser.parse_args()
    sigopt_tune(args)
    print('done!')
