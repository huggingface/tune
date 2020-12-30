from collections import defaultdict
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser


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

    # dfs = {filter(lambda path: path.startswith(framework), results.)}
    return dfs


def generate_report_by_seq_len(dfs, output_folder: Path):
    for framework, df in df_by_framework.items():
        df["framework"] = framework

    final_df = pd.concat(dfs.values())
    final_df["inference_time"] = final_df["inference_time"] * 1000

    sns.set_theme(style="ticks", color_codes=True)
    col_order = ["pytorch", "tensorflow", "torchscript", "tensorflow_xla"]

    # Plot data
    g = sns.catplot(
        x="seqlen",
        y="inference_time",
        hue="framework",
        data=final_df,
        kind="bar",
        height=6,
        aspect=2,
        margin_titles=True
    )

    # Title and legend
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Inference time for each framework (ms)', fontsize=16)
    g.set_xlabels("Sequence Length (nb tokens)")
    g.set_ylabels("Inference Time (ms)")
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("reports/inference_time_per_variable_args.svg", format="svg")
    plt.show()


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
        generate_report_by_seq_len(df_by_framework, args.output_folder)
    except ValueError as ve:
        print(ve)