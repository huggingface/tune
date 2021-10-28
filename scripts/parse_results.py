# Ref: Stackoverflow
# Usage: python get_results.py 

import os

# Model name - change this as needed
model_name = 'bert-base-cased'

# Main directory containing the results - change this as needed
rootdir = ''

results_file = model_name + '.csv'
with open(results_file, 'a') as wf:
    wf.write('id,nb_forwards,throughput,latency_mean (ms),latency_std (ms),latency_50 (ms),latency_90 (ms),latency_95 (ms),latency_99 (ms),latency_999 (ms),model,backend,seq_len,batch_size,num_threads\n')

# os.walk through the main results directory to change into each subdirectory, extract the perf results and append to the results file
for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        if filename.endswith(".csv"):
            file = os.path.join(subdir, filename)
            with open(file, 'r') as f:
                lines = f.readlines()
                #print(lines[1])
                with open(results_file, 'a') as wf:
                    wf.write(lines[1])
