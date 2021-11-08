#!/usr/bin/env python
# Copyright (c) 2014, Intel Corporation.

import sys, os
from argparse import ArgumentParser
import pandas as pd

def build_argparser():

    usage = '''example:
     python inter_results.py -input '/path/to/root/dir of benchmark CSV files' 
     --dirresults <optional: path to output dir to save summary_results.csv>
     '''

    parser = ArgumentParser(prog='inter_results.py',
                            description='Intermediate benchmark results',
                            epilog=usage)
    args = parser.add_argument_group('Options')
    args.add_argument('-input', '--dirpath', help='Path to root directory of csv files with model performance data', required=True)
    args.add_argument('-out', '--dirresults', help='Output results summary file .csv', required=False, type=str, default='result_summary.csv')

    return parser

def main():
    
    args = build_argparser().parse_args()

    df_ = pd.DataFrame()

    if not os.path.isdir(args.dirpath):
        print("Error: Invalid root directory")
        return -1
    else:
        root_path = args.dirpath

    for curr_dir, list_dirs, file_names in os.walk(root_path):
        for f in file_names:
            curr_dir_split = os.path.normpath(curr_dir).split(os.path.sep)
            if len(curr_dir_split) == 7:
                instance = curr_dir_split[-3]
            else:
                instance = curr_dir_split[-2]
            f_ext = os.path.splitext(f)[-1].lower()
            if f_ext == ".csv":
               fn = f.split('-')
               f_csv = os.path.join(curr_dir, f)
               df_csv = pd.read_csv(f_csv)
               df_csv['instance_id'] = [int(instance)]
               df_ = df_.append(df_csv)
    for i, s in enumerate(fn):
        if i == 0:
           st = str('')
        if i > 0 and i < len(fn) - 1:
           st = st + s
        if i >= 1   and i < len(fn) - 2:
           st += '_'
    df_.to_csv(str(st)+".csv", encoding='utf-8', index=False)

if __name__ == '__main__':
    sys.exit(main() or 0)
