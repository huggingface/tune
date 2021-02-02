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

from collections import defaultdict
from pathlib import Path

import pandas as pd
from argparse import ArgumentParser

from rich.console import Console
from rich.table import Table


def gather_results(folder: Path):
    # List all csv results
    results_f = [f for f in folder.glob("**/*.csv")]
    results_csv = {
        f.relative_to(folder).parent.as_posix(): pd.read_csv(f, index_col=0)
        for f in results_f
    }

    if len(results_csv) == 0:
        raise ValueError(f"No results.csv file were found in {folder}")

    # Merge dataframe wrt to framework
    dfs = defaultdict(list)
    for path, df in results_csv.items():
        framework, device, arguments = path.split("/")
        arguments = dict(arg.split("_") for arg in arguments.split("-"))

        # Add columns to the dataframe
        for col_name, col_value in arguments.items():
            df[col_name] = int(col_value)

        dfs[framework].append(df)

    # Concat the dataframes
    dfs = {f: pd.concat(a) for f, a in dfs.items()}

    for framework, df in dfs.items():
        df["framework"] = framework

    return pd.concat(dfs.values())


def show_results_in_console(df):
    grouped_df = df.groupby(["framework", "batch", "seqlen"])
    (grouped_df["inference_time_secs"].mean() * 1000).reset_index()

    console = Console()
    table = Table(
        show_header=True, header_style="bold",
        title="Inference Time per Framework, Batch Size & Sequence Length"
    )

    columns = (
        ("Framework", "framework"),
        ("Batch Size", "batch"),
        ("Seq Length", "seqlen"),
        ("Inference Time (ms)", "inference_time_secs")
    )

    # Define the columns
    for (column, _) in columns:
        table.add_column(column, justify="center")

    # Add rows
    for name, group in grouped_df:
        items = name + (round(group.mean()["inference_time_secs"] * 1000, 2), )
        table.add_row(*[str(item) for item in items])

    # Display the table
    console.print(table)


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Model Benchmark")
    parser.add_argument("--results-folder", type=Path, help="Where the benchmark results have been saved")
    parser.add_argument("output_folder", type=Path, help="Where the resulting report will be saved")

    # Parse command line arguments
    args = parser.parse_args()

    if not args.results_folder.exists():
        print(f"Folder {args.results_folder} doesn't exist")

    try:
        # Ensure output folder exists
        args.output_folder.mkdir(exist_ok=True, parents=True)

        # Gather the results to manipulate
        df_by_framework = gather_results(args.results_folder)

        # Generate reports
        df_by_framework.to_csv(args.output_folder.joinpath("final_results.csv"))

        show_results_in_console(df_by_framework)
    except ValueError as ve:
        print(ve)