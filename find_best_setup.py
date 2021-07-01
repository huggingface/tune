import sys
from argparse import ArgumentParser

from hpo.utils import TuningMode
from hpo.optuna import optuna_tune
from hpo.sigopt import sigopt_tune


def get_args():
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
        required="--convert_csv" not in sys.argv,
        help="The framework to evaluate",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=None,
        help="The number of instances to run the model on, if not specified, this parameter will be tuned",
    )
    parser.add_argument(
        "--openmp",
        type=str,
        default=None,
        choices=["openmp", "iomp"],
        help="The shared memory multiprocessing library to use, if not specified, this parameter will be tuned",
    )
    parser.add_argument(
        "--allocator",
        type=str,
        default=None,
        choices=["default", "tcmalloc", "jemalloc"],
        help="The allocator to use, if not specified, this parameter will be tuned",
    )
    parser.add_argument(
        "--huge_pages",
        type=str,
        default=None,
        choices=["on", "off"],
        help="Wether to use transparent huge pages, if not specified, this parameter will be tuned",
    )
    parser.add_argument(
        "--mode",
        type=TuningMode,
        choices=[
            TuningMode.LATENCY,
            TuningMode.THROUGHPUT,
            TuningMode.BOTH,
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
    parser.add_argument("--proxy", type=str, default=None, help="Proxies for sigopt")
    parser.add_argument("--logfile", type=str, default="tune.log", help="Log file")
    parser.add_argument("--convert_csv", action="store_true", help="Convert json log file to csv")
    parser.add_argument("--tuner", type=str, choices=["optuna", "sigopt"], help="The hyperparameter tuner to use")

    return parser.parse_args()


def args_to_parameters(args):
    main_names = {"batch_size", "sequence_length", "framework"}
    launcher_names = {"instances", "openmp", "allocator", "huge_pages"}
    main_parameters = {}
    launcher_parameters = {}
    other_parameters = {}
    for name in vars(args):
        arg = getattr(args, name)
        if not arg:
            continue
        if name in main_names:
            main_parameters[name] = arg
        elif name in launcher_names:
            launcher_parameters[name] = arg
        else:
            other_parameters[name] = arg

    main_parameters["backend"] = main_parameters["framework"]
    main_parameters.pop("framework")

    return main_parameters, launcher_parameters, other_parameters

if __name__ == "__main__":
    args = get_args()

    main_parameters, launcher_parameters, other_parameters = args_to_parameters(args)

    if args.tuner == "optuna":
        print("Starting hyperparameter tuning with Optuna")
        optuna_tune(
            main_parameters=main_parameters,
            launcher_parameters=launcher_parameters,
            **other_parameters,
        )

    else:
        print("Starting hyperparameter tuning with Sigopt")
        sigopt_tune(
            main_parameters=main_parameters,
            launcher_parameters=launcher_parameters,
            **other_parameters,


        )
