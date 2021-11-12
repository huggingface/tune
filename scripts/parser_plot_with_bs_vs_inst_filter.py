#!/usr/bin/env python
# Copyright (c) 2014, Intel Corporation.

import pandas as pd
import sys, os
from argparse import ArgumentParser, SUPPRESS

NUM_OF_INSTANCE = [1, 2, 3, 4]  # Mapped values of (1, 2, 4, 8)
SEQ_LENGTH = [20, 32, 128, 384, 512]
#BATCH_SIZE = [1, 2, 4, 8]
BATCH_SIZE = [8, 4, 2, 1]
#NUM_THREADS = [8, 16, 32, 64]
BACK_END = ['pytorch', 'openvino', 'tensorflow']

def build_argparser():

    usage = '''example:
     python benchark_filter.py -input '/path/to/dir with benchmark CSV files' 
          --dirresults <optional: path to output dir to save summary_results.csv>
     '''
    
    parser = ArgumentParser(prog='benchmark_filter.py',
                            description='Filter benchmark results',
                            epilog=usage)
    args = parser.add_argument_group('Options')
    args.add_argument('-input', '--dirpath', help='Benchmark csv files direcotry path', required=True)
    args.add_argument('-out', '--dirresults', help='Output results summary file .csv', required=False, type=str, default='result_summary.csv')
    
    return parser

def plot_results(summary_df):
     
    plot_df = pd.DataFrame()
    for seq_len in SEQ_LENGTH:
        for idx, no_ins in enumerate(NUM_OF_INSTANCE):
            #for bs in BATCH_SIZE:
                df = summary_df.loc[(summary_df['batch_size'] == BATCH_SIZE[idx]) & (summary_df['seq_len'] == seq_len) & \
                                       (summary_df['instance_id'] == no_ins)]
              
                #print("df.head()", df.head())
                #sys.exit(0)
                if not df.empty:
                   back_end = df.backend.unique()
                   #df_temp = pd.DataFrame()
                   df_ = pd.DataFrame(columns=['sl','inst','bs','threads','pytorch','openvino'])
                   ov_l = []
                   pt_l = []
                   num_threads = []
                   for index, row in df.iterrows():
                       if row.loc['backend'] in BACK_END:
                           if row.backend == 'openvino':
                               ov_latency = row.loc['latency_mean (ms)']
                           elif row.backend == 'pytorch':
                               pt_latency = row.loc['latency_mean (ms)']

                   df_.loc[df_.index.max() + 1, :] = [row.loc['seq_len'], row.loc['instance_id'], row.loc['batch_size'], \
                                                           row.loc['num_threads'], pt_latency, ov_latency]
                        #df_.loc[:'inst'] = row.loc['instance_id']
                        #df_.loc[:'bs'] = row.loc['batch_size']
                        #df_.loc[:'threads'] = row.loc['num_threads']
                        #df_.loc[:'pytorch'] = pt_latency
                        #df_.loc[:'openvino'] = ov_latency
                        #print("df_ ==", df_)
                        #df_temp = df_temp.append(df_)
                        #print("df_temp ==", df_temp)
                   plot_df = plot_df.append(df_)
    plot_df = plot_df.drop_duplicates()
    plot_df.to_csv("plot_results.csv", encoding='utf-8', index=False)

def main():
    args = build_argparser().parse_args()

    if not os.path.isdir(args.dirpath):
        print("ERROR: Direcotry not found..")
        return -1
    if not os.listdir(args.dirpath):
        print("ERROR: Empty file found in args.dirpath")
        return -1
        
    summary_df = pd.DataFrame()
    for file_name in os.listdir(args.dirpath):
        csv = os.path.isfile(os.path.join(args.dirpath, file_name))
        #print("file = ", file_name)
        f_n = os.path.splitext(file_name)[0]
        #print("f_n = ", f_n)
        f_ext = os.path.splitext(file_name)[-1].lower()
        if f_ext == ".csv" and csv:
        #if csv:
            df = pd.read_csv(os.path.join(args.dirpath, file_name))
            back_end = df.backend.unique()
            seq_len = df.seq_len.unique()
            batch_size = df.batch_size.unique()
            num_threads = df.num_threads.unique()
            num_instance = df['instance_id'].unique()
            total_inst = len(num_instance)
            #print("total_inst = ", total_inst)
            # Backend
            for backend in back_end:
              #print("in loop backend", backend)
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
    
    plot_results(summary_df)

if __name__ == '__main__':
    sys.exit(main() or 0)
