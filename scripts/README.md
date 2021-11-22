## Benchmarking HF tune models

```bash
# Clone project tune:
git clone -b ov-dev https://github.com/karkadad/tune.git
```

```bash
# Launching benchmarking shell script 
./scripts/launch.sh
# Expected output: Benchmark results .csv files under "<project clone path>/tune/outputs/default/"
```

```bash
# Parsing results from Step 2 into single model csv file 
# Command line: $tune/scripts: 
python3 launch_results_parser.py -i <project clone path>/tune/outputs/default/
# Expected output: CSV file(s) with "<hf_model_name>.csv"
```

```bash
# Parsing "<hf_model_name>.csv" for benchmark plot
# Command Line $tune/scripts: 
python3 parser_with_selec_params.py -i <project clone path>/tune/scripts/.
# Expected output: csv file with formated results table for easy plotting and visualization
```
 
#### Note: To run GPT2/ROBERTA set below environment
```bash
# Set ENVs
TRANSF_PIP_LOC=`pip show transformers | grep Location | cut -d " " -f2`
export ROBERTA_FILE=$TRANSF_PIP_LOC/transformers/models/roberta/tokenization_roberta_fast.py
export GPT2_FILE=$TRANSF_PIP_LOC/transformers/models/gpt2/tokenization_gpt2_fast.py

# Run below to change flags
sed -i '134s/False/True/' $GPT2_FILE
sed -i '157s/False/True/' $ROBERTA_FILE
```

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
  TRANSF_PIP_LOC=`pip show transformers | grep Location | cut -d " " -f2`
  export ROBERTA_FILE=$TRANSF_PIP_LOC/transformers/models/roberta/tokenization_roberta_fast.py
  export GPT2_FILE=$TRANSF_PIP_LOC/transformers/models/gpt2/tokenization_gpt2_fast.py
  ```
  
  Expected Output:
  Benchmark details of multiple models with multi-insatnce(s) saved in .csv files under "outputs/default" root path 
  
## Consolidating results generated from "launch.sh"
  
  Python script "launch_results_parser.py" to segregate benchmark results w.r.t model vs backend vs seq_len vs batch vs num_instances
  
 
  
  Expected output file:
   Ex: For "bert_base_cased" HF model. A file with "bert_base_cased.csv" will be generated as shown below
   
| id	| nb_forwards |	throughput  | latency_mean (ms)| .. | model           | backend | seq_len | bs  |	threads |	instance |
| --- |:-----------:|:-----------:|:----------------:|:--:|:---------------:|:-------:|:-------:|:---:|:-------:| --------:|
| 0   |	753	        |	37.65       | 26.5818909	     | .. | bert-base-cased |	pytorch	| 128	    |  2	|  32     |	1        |
| 1   |	1381        |	69.05       | 14.48825. 	     | .. | bert-base-cased |	pytorch	| 32	    |  2	|  32     |	1        |


## Benchmark visualization

### Extracting required fields 

Python script "parser.py" does extract required fields (as shown below) for plotting benchmark results

Expected output results_summary.csv 

|throughput |	latency_mean (ms) |	instance_id	|    seq_len	     | bs |	num_threads |	model_name      | backend  |
| --------- |:-----------------:|:-----------:|:----------------:|:--:|:-----------:|:---------------:|:--------:|
|  75.45	  |    13.25840211 	  |      1	    |         20       |	2 |	    32	    | bert_base_cased	|  pytorch | 


### Arranging rows/columns for data visulization

#### Bar-chart with "num_threads, num_instances, batch_size, seq_len" (x-axis) vs "latench in (msec)"

 Use "parser_with_selec_params.py" to arrange data to plot bar chart with above mentioned x-axis variables vs latency (msec) data.
 
#### Arranging data (all combinations of attributes) for plotting
 
 Use "parser_plot_with_all_params.py" to arrange data with all combination of params
 
 
   
