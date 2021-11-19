#!/bin/bash
# Copyright (c) 2019 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Pradeep, Sakhamoori <pradeep.sakhamoori@intel.com>
# Date : 11/10/2021

import pandas as pd
import sys, os
from argparse import ArgumentParser, SUPPRESS

def build_argparser():

    usage = '''example:
     python parser.py -i '/path/to/dir with benchmark CSV files' 
          --o <optional: path to output dir to save summary_results.csv>
     '''
    
    parser = ArgumentParser(prog='parser.py',
                            description='Filter benchmark results',
                            epilog=usage)
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--input_dir', help='Benchmark csv files direcotry path', required=True)
    args.add_argument('-o', '--output_dir', help='Output results summary file .csv', required=False, type=str, default='result_summary.csv')
    
    return parser

def main():
    args = build_argparser().parse_args()

    if not os.path.isdir(args.input_dir):
        print("ERROR: Direcotry not found..")
        return -1
    if not os.listdir(args.input_dir):
        print("ERROR: No files found...")
        return -1
        
    summary_df = pd.DataFrame()
    for file_name in os.listdir(args.input_dir):
        csv = os.path.isfile(os.path.join(args.input_dir, file_name))
        print("file = ", file_name)
        f_n = os.path.splitext(file_name)[0]
        #print("f_n = ", f_n)
        f_ext = os.path.splitext(file_name)[-1].lower()
        if f_ext == ".csv" and csv:
        #if csv:
            df = pd.read_csv(os.path.join(args.input_dir, file_name))
            back_end = df.backend.unique()
            seq_len = df.seq_len.unique()
            batch_size = df.batch_size.unique()
            num_threads = df.num_threads.unique()
            num_instance = df['instance_id'].unique()
            total_inst = len(num_instance)
            print("total_inst = ", total_inst)
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
                          df_bs = df_bs[['instance_id', 'throughput', 'latency_mean (ms)', 'seq_len', 'batch_size', 'num_threads']]
                          df_bs_m = df_bs.mean(axis=0)
                          summary_temp = summary_temp.append(df_bs_m, ignore_index=True)
                          summary_temp['model_name'] = f_n
                          summary_df = summary_df.append(summary_temp.assign(backend=backend))
    summary_df = summary_df.drop_duplicates()
    summary_df.to_csv("results_summary.csv", encoding='utf-8', index=False)
                
if __name__ == '__main__':
    sys.exit(main() or 0)
