from pathlib import Path

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
    frameworks = {path.split("/")[0] for path in results_csv.keys()}

    # dfs = {filter(lambda path: path.startswith(framework), results.)}
    return dfs


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Model Benchmark")
    parser.add_argument("folder", type=Path)

    # Parse command line arguments
    args = parser.parse_args()

    if not args.folder.exists:
        print(f"Folder {args.folder} doesn't exist")

    try:
        # Gather the results to manipulate
        results = gather_results(args.folder)
    except ValueError as ve:
        print(ve)