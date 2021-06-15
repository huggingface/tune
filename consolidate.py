#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from datetime import datetime, timezone
from glob import glob
from itertools import chain
from os import path
from pathlib import Path
from typing import Type, List, Tuple

import pandas as pd
from argparse import ArgumentParser

import yaml
from pandas import ExcelWriter
from rich.console import Console
from rich.table import Table

# Format name -> extension
SUPPORTED_EXPORT_FORMAT = {
    "csv": "csv",
    "excel": "xlsx"
}


SCALING_CHOICES = {"batch-size-scaling", "core-count-scaling"}
SCALING_HELP = "Which scaling metodology was used:\n \
                \t- batch-size-scaling: The total number of cores for the original batch size remains the same - \
                we use all the cores for the given batch size but break up the problem into smaller problems \
                with fewer cores for the smaller problem sizes\n" \
                "\t core-count-scaling: We vary the number of cores for the given batch size"


LATENCY_COLUMNS = {
    "latency_mean",
    "latency_std",
    "latency_50",
    "latency_90",
    "latency_95",
    "latency_99",
    "latency_999",
}

LATENCY_THROUGHPUT_COLUMNS = {
    "throughput",
}.union(LATENCY_COLUMNS)


SUMMARY_SUMMING_COLUMNS = {
    "nb_forwards",
    "throughput",
    "batch_size",
}

FINAL_COLUMNS_ORDERING = ["backend.name", "batch_size", "sequence_length", "openmp.backend", "malloc", "use_huge_page"]
RICH_DISPLAYED_COLUMNS = {
    "backend.name": "Backend",
    "malloc": "Malloc",
    "openmp.backend": "OpenMP",
    "use_huge_page": "Huge Pages",
    "batch_size": "Batch",
    "sequence_length": "Sequence",
    "latency_mean": "Avg. Latency",
    "latency_std": "Std. Latency",
    "throughput": "Throughput",
    "num_core_per_instance": "Cores"
}

MULTI_INSTANCES_VALIDATION_COLUMNS = [
    "batch_size",
    "sequence_length",
    "backend.name",
    "openmp.backend",
    "malloc",
    "backend.num_threads",
    "use_huge_page"
]


def flatten_yaml(path: Path, loader: Type[yaml.Loader] = yaml.SafeLoader) -> pd.DataFrame:
    with open(path, "r") as yaml_f:
        content = yaml.load(yaml_f, Loader=loader)

    return pd.json_normalize(content)


def gather_results(folder: Path, is_multi_instances: bool) -> Tuple[pd.DataFrame, List[str]]:
    # List all csv results
    results_f = [(f, f.parent.joinpath(".hydra/config.yaml")) for f in folder.glob("**/results.csv")]
    results_df = pd.concat([
        # This will concatenate columns from the benchmarks along with config columns
        pd.concat((pd.read_csv(results, index_col=0), flatten_yaml(config)), axis="columns")
        for results, config in results_f
    ], axis="index")

    existing_columns = list(set(FINAL_COLUMNS_ORDERING).intersection(results_df.columns))
    results_df = results_df.sort_values(existing_columns)

    # Ensure the number of instances (according to the sum of instance_sum) matchs num_instances field
    if is_multi_instances:
        results_df["is_valid"] = results_df.groupby(MULTI_INSTANCES_VALIDATION_COLUMNS)["instance_id"].transform("count")
        results_df["is_valid"] = results_df["is_valid"] == results_df["num_instances"]
    else:
        results_df["is_valid"] = True

    results_df.fillna("N/A", inplace=True)
    if len(results_df) == 0:
        raise ValueError(f"No results.csv file were found in {folder}")

    return results_df, existing_columns


def aggregate_multi_instances_results(results_df: pd.DataFrame, grouping_columns: List[str], mode: str):
    agg_df = results_df.copy()
    agg_df = agg_df.groupby(grouping_columns)
    transforms = {
        "latency_mean": ["min", "max", "mean"],
        "throughput": ["sum"],
        "is_valid": ["all"]
    }

    # How to aggregate cores and batch
    if mode == "batch-size-scaling":
        transforms["batch_size"] = "sum"

    return agg_df.agg(transforms)


def show_results_in_console(df: pd.DataFrame, sorting_columns: List[str]):
    console = Console(width=200)
    table = Table(
        show_header=True, header_style="bold",
        title="Latency & Throughput for each framework (latencies given in ms)",
    )

    # Create copy
    local_df = df.copy()
    local_df = local_df.assign(**local_df[LATENCY_COLUMNS].apply(lambda x: round((x * 1e-6), 2)))

    # Filter out columns
    displayed_columns = {
        column_id: column_title
        for column_id, column_title in RICH_DISPLAYED_COLUMNS.items()
        if column_id in local_df.columns
    }

    for column_name in displayed_columns.values():
        table.add_column(column_name, justify="center")
    table.add_column("Instance ID", justify="center")

    # Add rows
    for _, item_columns in local_df.sort_values(sorting_columns, ascending=True).iterrows():
        table.add_row(*[str(item_columns[c]) for c in chain(displayed_columns.keys(), ["instance_id"])])

    # Display the table
    console.print(table)


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Model Benchmark")
    parser.add_argument("--results-folder", type=Path, help="Where the benchmark results have been saved")
    parser.add_argument("--multi-instances-scaling", choices=SCALING_CHOICES, help=SCALING_HELP)
    parser.add_argument("--format", choices=SUPPORTED_EXPORT_FORMAT.keys(), default="csv", help="Export file format")
    parser.add_argument("output_folder", type=Path, help="Where the resulting report will be saved")

    # Parse command line arguments
    args = parser.parse_args()
    args.now = datetime.now(timezone.utc).astimezone()
    args.experiment_id = path.split(args.results_folder)[-1]
    args.format_ext = SUPPORTED_EXPORT_FORMAT[args.format.lower()]

    for name in {"aggregated", "consolidated"}:
        value = f"{name}_{args.experiment_id}_" \
                f"{args.now.date().isoformat()}T{args.now.time().strftime('%H-%M')}" \
                f".{args.format_ext}"
        setattr(args, f"{name}_filename", value)

    # Ensure everything looks right
    if not args.results_folder.exists():
        print(f"Folder {args.results_folder} doesn't exist")
        exit(1)

    try:
        # Detect folder run type from folder structure
        instances_folder = glob(f"{args.results_folder.as_posix()}/*")

        args.is_multi_instances = len(instances_folder) > 1
        args.instances = {path.split(instance_folder)[-1] for instance_folder in instances_folder}
        args.is_multirun = {
            path.split(instance_folder)[-1]: path.exists(path.join(instance_folder, "multirun.yaml"))
            for instance_folder in instances_folder
        }

        print(
            f"Detected following structure:"
            f"\n\t- Multi Instance: {args.is_multi_instances} ({len(args.instances)} instances)"
            f"\n\t- Multirun: {args.is_multirun}"
          )

        # If we detect multi instance and no scaling mode is provided, ask for a value
        if args.is_multi_instances and args.multi_instances_scaling is None:
            print(
                "Warning:\n\tNo mode for handling multi-instances aggregation was provided. "
                "Only individual runs will be saved.\n"
                "\tTo include multi-instances aggregation results, "
                f"please use --multi-instance-scaling={SCALING_CHOICES}\n"
          )

        # Ensure output folder exists
        args.output_folder.mkdir(exist_ok=True, parents=True)

        # Gather the results to manipulate
        consolidated_df, sorting_columns = gather_results(args.results_folder, args.is_multi_instances)

        if args.is_multi_instances and args.multi_instances_scaling is not None:
            agg_df = aggregate_multi_instances_results(consolidated_df, sorting_columns, args.multi_instances_scaling)

        if args.format == "csv":
            consolidated_df.to_csv(args.output_folder.joinpath(args.consolidated_filename))
            if args.is_multi_instances and args.multi_instances_scaling is not None:
                agg_df.to_csv(args.output_folder.joinpath(args.aggregated_filename))
        else:
            with ExcelWriter(args.output_folder.joinpath(args.consolidated_filename)) as excel_writer:
                consolidated_df.to_excel(excel_writer, sheet_name="individuals")
                if args.is_multi_instances and args.multi_instances_scaling is not None:
                    agg_df.to_excel(excel_writer, sheet_name="aggregated_multi_instances")

        show_results_in_console(consolidated_df, sorting_columns)
    except ValueError as ve:
        print(ve)