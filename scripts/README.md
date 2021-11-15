## HF tune model(s) benchmarking 
  Shell script "launch.sh" takes in below list of pre-populated variables and generates "outputs/default" list of folders
  with benchmarking data saved in .csv files 

  Input varaibles (with default settings):

  ```bash
  Backend framework(s): BACKEND=pytorch,ov,tensorflow.
  Inference batch size: BATCH_SIZE=1,2,4,8.
  Text sequence length: SEQ_LEN=20,32,128,384,512.
  Number of inference instance(s): N_INSTS=(1 2 4 8).
  List of HF models: MODELS=(bert-base-cased distilbert-base-uncased gpt2).
  ```
 
  Export environment variables:
  
  ```bash
  export ROBERTA_FILE=</path/to/lib/pythonx.x>/dist-packages/transformers/models/roberta/tokenization_roberta_fast.py
  export GPT2_FILE=</path/to/lib/pythonx.x>dist-packages/transformers/models/gpt2/tokenization_gpt2_fast.py
  ```
  
  Expected Output:
  Benchmark details of multiple models with multi-insatnce(s) saved in .csv files under "outputs/default" root path 
  
## Consolidating results generated from "launch.sh"
  
  Python script "launch_results_parser.py" to segregate benchmark results w.r.t model vs backend vs seq_len vs batch vs num_instances
  
  ```bash
     ubuntu@ip:~/tune/scripts$ python3 inter_results.py --help
     usage: inter_results.py [-h] -input DIRPATH [-out DIRRESULTS]
     Intermediate benchmark results

     optional arguments:
      -h, --help            show this help message and exit

    Options:
      -input DIRPATH, --dirpath DIRPATH
                        Path to root directory of csv files with model performance data
      -out DIRRESULTS, --dirresults DIRRESULTS
                          Output results summary file .csv

   Example: python3 inter_results.py -input </path/to/root/dir/ of benchmark csv data> 
                                     --dirresults <optional: path to output dir to save summary_results.csv>
  ```
  
  Expected output file:
   Ex: For "bert_base_cased" HF model. A file with "bert_base_cased.csv" will be generated as shown below
   
| id	| nb_forwards |	throughput  | latency_mean (ms)| .. | model           | backend | seq_len | bs  |	threads |	instance |
| --- |:-----------:|:-----------:|:----------------:|:--:|:---------------:|:-------:|:-------:|:---:|:-------:| --------:|
| 0   |	753	        |	37.65       | 26.5818909	     | .. | bert-base-cased |	pytorch	| 128	    |  2	|  32     |	1        |
| 1   |	1381        |	69.05       | 14.48825. 	     | .. | bert-base-cased |	pytorch	| 32	    |  2	|  32     |	1        |


## Benchmark visualization

### Extracting required fields 

Python script "parser.py" does extract required fields (as shown below) for plotting benchmark results

```bash
  usage = '''example:
     python parser.py -input '/path/to/dir with benchmark CSV files' 
          --dirresults <optional: path to output dir to save summary_results.csv>
     '''
    
    parser = ArgumentParser(prog='parser.py',
                            description='Filter benchmark results',
                            epilog=usage)
    args = parser.add_argument_group('Options')
    args.add_argument('-input', '--dirpath', help='Benchmark csv files direcotry path', required=True)
    args.add_argument('-out', '--dirresults', help='Output results summary file .csv', required=False, type=str, default='result_summary.csv')
 ```
Expected output results_summary.csv 

|throughput |	latency_mean (ms) |	instance_id	|    seq_len	     | bs |	num_threads |	model_name      | backend  |
| --------- |:-----------------:|:-----------:|:----------------:|:--:|:-----------:|:---------------:|:--------:|
|  75.45	  |    13.25840211 	  |      1	    |         20       |	2 |	    32	    | bert_base_cased	|  pytorch | 


### Arranging rows/columns for data visulization

#### Bar-chart with "num_threads, num_instances, batch_size, seq_len" (x-axis) vs "latench in (msec)"

 Use "parser_with_selec_params.py" to arrange data to plot bar chart with above mentioned x-axis variables vs latency (msec) data.
 
#### Arranging data (all combinations of attributes) for plotting
 
 Use "parser_plot_with_all_params.py" to arrange data with all combination of params
 
 
   
