## Transformers benchmarks in collaboration with Intel Corp.

The benchmarking repository provides an easy and flexible testbed to generate, run and save multiple configurations 
in order to compare Transformers based Neural Network models.

The overall benchmarking project leverage the framework Hydra from Facebook AI & Research which is able to generate
all the sweeps given through configurations files. Currently, we provide benchmarks for 5 Deep Learning frameworks
among the most used: 

- PyTorch (Eager mode)
- TorchScript (Static Graph mode)
- TensorFlow 2 (Eager mode)
- TensorFlow 2 XLA (Static Graph mode)
- ONNX Runtime for Inference (Static Graph mode + Graph Optimizations)

The repository is divided into 2 principal sections:
- `config/` stores all the configuration files for the supported backends.
- `backends/` stores the actual logic to generate textual inputs and execute a forward pass for the targeted backend.


## How to use this repository to benchmark with a specific configuration

Hydra, the configuration framework used in this project, provides a simple command-line interface to specify and
override the configuration to be run.

For instance, in order to run a benchmark for ONNX Runtime on CPU with:
- **Backend = ORT**
- **Model = bert-base-cased**
- **Device = CPU**
- **Batch Size = 1**
- **Sequence Length = 32**

```bash
python3 src/main.py model=bert-base-cased sequence_length=32 backend=ort device=cpu
```

## Automatically let Hydra generate all the permutation to cover multiple configurations

Hydra integrates a very powerfull sweep generation utility which is exposed through the `--multirun` command-line flag
when invoking the benchmark script.

For instance, in order to run multiple benchmarks for PyTorch/TensorFlow/ONNX Runtime on CPU with:
- **Model = bert-base-cased**
- **Device = CPU**
- **Batch Size = 1,4,8**
- **Sequence Length = 8,16,32,64,128,256,512**

```bash
python3 src/main.py --multirun model=bert-base-cased batch_size=1,4,8 sequence_length=8,16,32,64,128,256,512 backend=pytorch,tensorflow,ort device=cpu
```

## Overridable configuration properties

- `backend`: Indicate the backend(s) to use to run the benchmark `{"pytorch", "torchscript", "tensorflow", "xla", "ort"}`
- `device`: Indicate on which device to run the benchmark `{"cpu", "gpu"}`
- `precision`:
- `num_threads`: Number of threads to use for intra-operation within OpenMP parallel section (`-1` Detect the number of CPU cores and use this value)
- `num_interops_threads`: Number of threads to use for inter-operation within OpenMP parallel section (`-1` Detect the number of CPU cores and use this value)
- `warmup_runs`: Number of warmup forward to execute before recording any benchmarking results. (Especially useful to preallocate memory buffers).
- `num_runs`: Number of forward call to execute to collect benchmarking results. These runs are executed after `warmup_runs`.


## Backend specific configuration properties

Framework expose different features which can be enabled to tune the execution of the model on the underlying hardware.
In this repository we expose some of them, essentially the most common one.

### PyTorch

- `use_torchscript` Boolean indicating if the runtime should trace the eager model to produce an optimized version.

This value is `False` when using backend `pytorch` and `True` when using backend `torchscript` 

### TensorFlow

- `use_xla` Boolean indicating if the model should be wrapped around `tf.function` in order to compile the underlying graph through XLA.

This value is `False` when using backend `tensorflow` and `True` when using backend `xla`


### ONNX Runtime (ORT)

- `opset` Integer setting which version of the ONNX Opset specification to use when exporting the model

- `graph_optimisation_level` Which level of optimization to apply with ONNX Runtime when loading the model. Possible values are:
   - `ORT_DISABLE_ALL` Use the raw ONNX graph without any further optimization.
   - `ORT_ENABLE_BASIC` Use basic graph optimizations which are not platform dependant.
   - `ORT_ENABLE_EXTENDED` Use more advanced technics *(might include platform dependant optimizations)*.
   - `ORT_ENABLE_ALL` Enable all the possible optimizations *(might include platform dependant optimizations)*.
    
- `execution_mode` Mode to execute the ONNX Graph. Can be either:
   - `ORT_SEQUENTIAL` Execute the graph sequentially, without looking for subgraph to execute in parallel.
   - `ORT_PARALLEL` Execute the graph potentially in parallel, looking for non-dependant subgraphs which can be run simultaneously.
  

## Hydra FAQ 

### Overriding specific environment variables through Hydra's CLI

$ python my_app.py '+hydra.job.env_set={KMP_AFFINITY:granularity\=fine}' --cfg hydra -p hydra.job.env_set


### TODO 

### Machine configuration
- Open MP 
  - KMP_AFFINITY
  
- Intel Open MP
- Malloc libraries
  - tcalloc
    
- Transparent Huge pages
- Multi-instance runs
  - See FastFormer work
  - taskset / numactl (slide 22.)

- Dynamic sequence length for Batch Size = 1

### Model configuration  

- stick to BERT base cased 
- sequence length:
- batch size: 
- 
    
## Results 

We provide some results as part of this repository for easy access. 

- **Intel Core i9 10980XE CPU**


| `num_threads` | `num_interops_threads` | `device`   | pytorch | tensorflow | onnxruntime |  
|:-------------:|:----------------------:|:----------:|:-------:|:----------:|:-----------:|
|**18 (set to -1)** | **18 (set to -1)** |**`"cpu"`** | 1.7.1   | 2.4.0      | 1.6.0       |

| Batch Size | Sequence Length | PyTorch     | TorchScript | TensorFlow  | XLA         | ONNX Runtime |
|:----------:|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| 1          | 8               | 15.841 ms   | 11.075 ms   | 83.836 ms   | 25.684 ms   | 6.53 ms      |
| 1          | 16              | 21.307 ms   | 15.518 ms   | 176.191 ms  | 37.869 ms   | 7.388 ms     |
| 1          | 32              | 24.119 ms   | 18.244 ms   | 212.827 ms  | 44.938 ms   | 10.186 ms    |
| 1          | 64              | 28.468      | 22.975 ms   | 258.155 ms  | 74.185 ms   | 15.364 ms    |
| 1          | 128             | 36.843 ms   | 29.654 ms   | 329.318 ms  | 107.549 ms  | 26.618 ms    |
| 1          | 256             | 56.368 ms   | 49.459 ms   | 391.055 ms  | 200.078 ms  | 57.504 ms    |
| 1          | 512             | 115.465 ms  | 136.691 ms  | 533.576 ms  | 395.334 ms  | 114.748 ms   |

| Batch Size | Sequence Length | PyTorch     | TorchScript | TensorFlow  | XLA         | ONNX Runtime |
|:----------:|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| 4          | 8               | 24.633 ms   | 18.695 ms   | 201.572 ms  | 44.286 ms   | 14.409 ms    |
| 4          | 16              | 28.079 ms   | 22.369 ms   | 248.334 ms  | 61.281 ms   | 18.678 ms    |
| 4          | 32              | 35.426 ms   | 29.28 ms    | 311.369 ms  | 84.266 ms   | 27.617 ms    |
| 4          | 64              | 53.029 ms   | 45.386 ms   | 371.776 ms  | 172.184 ms  | 47.127 ms    |
| 4          | 128             | 85.482 ms   | 76.687 ms   | 463.447 ms  | 275.207 ms  | 90.433 ms    |
| 4          | 256             | 159.524 ms  | 184.298 ms  | 868.517 ms  | 555.863 ms  | 171.662 ms   |
| 4          | 512             | 413.618 ms  | 537.27 ms   | 1399.93 ms  | 1185.992 ms | 354.402 ms   |

| Batch Size | Sequence Length | PyTorch     | TorchScript | TensorFlow  | XLA         | ONNX Runtime |
|:----------:|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| 8          | 8               | 28.743 ms   | 22.431 ms   | 241.052 ms  | 61.94 ms    | 18.167 ms    |
| 8          | 16              | 36.084 ms   | 28.783 ms   | 306.034 ms  | 79.259 ms   | 26.897 ms    |
| 8          | 32              | 52.939 ms   | 45.286 ms   | 365.816 ms  | 147.997 ms  | 47.752 ms    |
| 8          | 64              | 82.7 ms     | 74.688 ms   | 455.898 ms  | 257.108 ms  | 88.991 ms    |
| 8          | 128             | 150.072 ms  | 126.975 ms  | 545.884 ms  | 481.025 ms  | 162.648 ms   |
| 8          | 256             | 342.636 ms  | 369.131 ms  | 1125.268 ms | 1159.149 ms | 270.288 ms   |
| 8          | 512             | 1063.939 ms | 1078.999 ms | 2410.667 ms | 2364.330 ms | 648.959 ms   |