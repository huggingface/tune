#!/usr/bin/env python
# Copyright (c) 2014, Intel Corporation.

import pandas as pd
import sys, os
from argparse import ArgumentParser, SUPPRESS

#NUM_OF_INSTANCE = [2, 4, 8]
#SQE_LENGTH = [20, 32, 128, 384, 512]
#BATCH_SIZE = [1, 2, 4, 8]
#NUM_THREADS = [8, 16, 32, 64]
#BACK_END = ["pytorch", "tensorflow", "openvino"]

def build_argparser():

    usage = '''example:
     python benchark_filter.py -input '/path/to/dir with benchmark CSV files' --dirresults <optional: path to output dir to save summary_results.csv>
     '''
    
    parser = ArgumentParser(prog='benchmark_filter.py',
                            description='Filter benchmark results',
                            epilog=usage)
    args = parser.add_argument_group('Options')
    args.add_argument('-input', '--dirpath', help='Benchmark csv files direcotry path', required=True)
    args.add_argument('-out', '--dirresults', help='Output results summary file .csv', required=False, type=str, default='result_summary.csv')
    
    return parser

def main():
    args = build_argparser().parse_args()

    if not os.path.isdir(args.dirpath):
        print("ERROR: Direcotry not found..")
        return -1
    if not os.listdir(args.dirpath):
        print("ERROR: Empty file found in args.dirpath")
        return -1
        
    summary_df = pd.DataFrame()
    for file in os.listdir(args.dirpath):
        csv = os.path.isfile(os.path.join(args.dirpath, file))
        print("file = ", file)
        if csv:
            df = pd.read_csv(os.path.join(args.dirpath, file))
            back_end = df.backend.unique()
            seq_len = df.seq_len.unique()
            batch_size = df.batch_size.unique()
            num_threads = df.num_threads.unique()
            num_instance = df[' instance_id'].unique()
            total_inst = len(num_instance)
            # Backend
            for backend in back_end:
              print("in loop backend", backend)
              df_backend = df.loc[df['backend'] == backend]
              # Sequnce length
              for sl in seq_len:
                  df_seq_len = df_backend.loc[df_backend['seq_len'] == sl]
                  # Numb of threads
                  for nt in num_threads:
                      df_num_threads = df_seq_len.loc[df_seq_len['num_threads'] == nt]
                      # Batch size
                      for bs in batch_size:
                          summary_temp = pd.DataFrame()
                          df_bs = df_num_threads.loc[df_num_threads['batch_size'] == bs]
                          df_bs = df_bs[[' instance_id', 'throughput', 'latency_mean (ms)', 'seq_len', 'batch_size', 'num_threads']]
                          summary_temp = summary_temp.append(df_bs.groupby(' instance_id', as_index=False).sum())
                          summary_temp = summary_temp.div(total_inst)
                          summary_temp[' instance_id'] = total_inst
                          summary_df = summary_df.append(summary_temp.assign(backend=backend))
    summary_df = summary_df.drop_duplicates()
    summary_df.to_csv("results_summary.csv", encoding='utf-8', index=False)
                
if __name__ == '__main__':
    sys.exit(main() or 0)
    
