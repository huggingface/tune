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
from pathlib import Path
from typing import Type

import pandas as pd
from argparse import ArgumentParser

import yaml
from rich.console import Console
from rich.table import Table


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


RICH_DISPLAYED_COLUMNS = {
    "backend",
    "batch_size",
    "sequence_length",
    "latency_mean",
    "latency_std",
    "latency_90",
    "latency_99",
    "latency_999",
    "throughput",
    "num_core_per_instance"
}


def flatten_yaml(path: Path, loader: Type[yaml.Loader] = yaml.SafeLoader) -> pd.DataFrame:
    with open(path, "r") as yaml_f:
        content = yaml.load(yaml_f, Loader=loader)

    return pd.json_normalize(content)


def gather_results(folder: Path) -> pd.DataFrame:
    # List all csv results
    results_f = [(f, f.parent.joinpath(".hydra/config.yaml")) for f in folder.glob("**/results.csv")]
    results_df = pd.concat([
        # This will concatenate columns from the benchmarks along with config columns
        pd.concat((pd.read_csv(results, index_col=0), flatten_yaml(config)), axis="columns")
        for results, config in results_f
    ], axis="index")

    # Let's use instance id as string
    results_df["instance_id"] = results_df["instance_id"].astype("str", copy=False)

    # Aggregate results for multi instances
    if results_df.shape[0] > 1:
        mean_results = results_df[LATENCY_COLUMNS].aggregate("mean")
        summed_results = results_df[SUMMARY_SUMMING_COLUMNS].sum()
        aggr_results = pd.concat((mean_results, summed_results))

        results_df = results_df.append(aggr_results, ignore_index=True)
        results_df.loc[results_df.shape[0] - 1, "instance_id"] = "\u03A3"

    results_df.fillna("N/A", inplace=True)
    if len(results_df) == 0:
        raise ValueError(f"No results.csv file were found in {folder}")

    return results_df


def show_results_in_console(df: pd.DataFrame):
    console = Console(width=200)
    table = Table(
        show_header=True, header_style="bold",
        title="Latency & Throughput for each framework (latencies given in ms)",
    )

    # Create copy
    local_df = df.copy()
    local_df = local_df.assign(**local_df[LATENCY_COLUMNS].apply(lambda x: round((x * 1e-6), 2)))

    # We can remove these columns because their correctly referred to in the "backend.name" column
    columns = ["instance_id"] + list(filter(lambda name: name in RICH_DISPLAYED_COLUMNS, local_df.columns))

    # Define the columns
    for column in columns:
        table.add_column(column.title(), justify="center")

    # Add rows
    for _, item_columns in local_df.sort_values("instance_id", ascending=False).iterrows():
        table.add_row(*[str(item_columns[c]) for c in columns])

    # Display the table
    console.print(table)


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Model Benchmark")
    parser.add_argument("--results-folder", type=Path, help="Where the benchmark results have been saved")
    parser.add_argument("output_folder", type=Path, help="Where the resulting report will be saved")

    # Parse command line arguments
    args = parser.parse_args()

    # Ensure everything looks right
    if not args.results_folder.exists():
        print(f"Folder {args.results_folder} doesn't exist")
        exit(1)

    try:
        # Ensure output folder exists
        args.output_folder.mkdir(exist_ok=True, parents=True)

        # Gather the results to manipulate
        consolidated_df = gather_results(args.results_folder)

        # Generate reports
        dt = datetime.now(timezone.utc).astimezone()
        consolidated_df.to_csv(
            args.output_folder.joinpath(
                f"consolidated_{dt.date().isoformat()}T{dt.time().strftime('%H-%M')}.csv"
            )
        )

        show_results_in_console(consolidated_df)
    except ValueError as ve:
        print(ve)