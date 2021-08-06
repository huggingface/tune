## Transformers performance & evaluation framework

The benchmarking repository provides an easy and flexible testbed to generate, run and save multiple configurations 
in order to compare Transformers based Neural Network models.

The overall benchmarking project leverages the Hydra framework from Facebook AI & Research which is able to generate
all the given sweeps through configurations files. Currently, we provide benchmarks for 5 Deep Learning frameworks
among the most used: 

- PyTorch (Eager mode)
- TorchScript (Static Graph mode)
- TensorFlow 2 (Eager mode)
- TensorFlow 2 Graph (Static Graph mode)
- ONNX Runtime for Inference (Static Graph mode + Graph Optimizations)

The repository is divided into 2 principal sections:
- `config/` stores all the configuration files for the supported backends.
- `backends/` stores the actual logic to generate textual inputs and execute a forward pass for the targeted backend.

## Getting Started

**Instructions presented here have been tested on Ubuntu 20.04**

```bash
apt update && apt -y install python3 python3-pip python3-dev libnuma-dev
cd <repo/path>
pip install -r requirements.txt
```


## Benchmarking framework
### How to use this repository to benchmark with a specific configuration

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

### Automatically let Hydra generate all the permutations to cover multiple configurations

Hydra integrates a very powerful sweep generation utility which is exposed through the `--multirun` command-line flag
when invoking the benchmark script.

For instance, in order to run a benchmark for PyTorch on CPU with the following specs:
- **Model = bert-base-cased**
- **Device = CPU**
- **Batch Size = 1**
- **Sequence Length = 128**

```bash
python3 src/main.py model=bert-base-cased batch_size=1 sequence_length=128 backend=pytorch device=cpu
```

### Overridable configuration properties

- `backend`: Specify the backend(s) to use to run the benchmark `{"pytorch", "torchscript", "tensorflow", "xla", "ort"}`
- `device`: Specify on which device to run the benchmark `{"cpu", "cuda"}`
- `precision`: Specify the model's parameters data format. For now, only supports `float32` (_i.e. full precision_)
- `num_threads`: Number of threads to use for intra-operation (`-1` Detect the number of CPU cores and use this value)
- `num_interops_threads`: Number of threads to use for inter-operation (`-1` Detect the number of CPU cores and use this value)
- `warmup_runs`: Number of warmup forward to execute before recording any benchmarking results. (Especially useful to preallocate memory buffers).
- `benchmark_duration`: Duration (in seconds) of the benchmark in an attempt to do as many forward calls as possible within the specified duration. These runs are executed after `warmup_runs`.

## Backend specific configuration properties

Framework exposes different features which can be enabled to tune the execution of the model on the underlying hardware.
In this repository we expose some of them, essentially the most common ones.

### PyTorch

- `use_torchscript` Boolean indicating if the runtime should trace the eager model to produce an optimized version.

This value is `False` when using backend `pytorch` and `True` when using backend `torchscript` 

### TensorFlow

- `use_xla` Boolean indicating if the model should be wrapped around `tf.function(jit_compile=True)` in order to compile the underlying graph through XLA.

This value is `False` when using backend `tensorflow_graph` and can be enabled by config file or cmd line.


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


## Launch utility tool
The benchmarking comes with a launcher tool highly inspired by [the one made available by Intel](https://github.com/intel/intel-extension-for-pytorch/blob/master/intel_pytorch_extension_py/launch.py).
The launcher tool helps you handle all the lower bits to configure experiments and get the best out of the platform you have.

More precisely, it will be able to configure the following elements:

- Linux transparent huge pages mechanism
- CPU cores affinity for OpenMP threads on NUMA platforms
- Memory affinity for OpenMP threads on NUMA platforms
- OpenMP configurations (KMP_AFFINITY, KMP_BLOCKTIME, OMP_NUM_THREADS, OMP_MAX_ACTIVE_LEVELS, etc.)
- Change at runtime the OpenMP library to be used (GNU / Intel)
- Change the memory allocation library to be used (std, tcmalloc, jemalloc)
- Setup multi-instances inference (multi independent models executing in parallel) with per-instance CPU core/memory affinity

The launcher script `launcher.py` is located at the root of transformers-benchmarks folder. 
You can run `python launcher.py --help` to get all the tuning options available.  

## Ready to use CLI command

### Benchmarking out of the box configuration for multiple backends
```shell
--multirun model=bert-base-cased backend=pytorch,torchscript,tensorflow,xla,ort
```

### Tuning the number of intra/inter ops for parallel sections (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.)

```shell
--multirun model=bert-base-cased batch_size=1 sequence_length=32 backend.num_threads=2,4,8 backend.num_interops_threads=2,4,8
```

### Tuning OpenMP thread affinity
```shell
python launcher.py --kmp_affinity=<value_here> -- src/main.py model=bert-base-cased batch_size=1 sequence_length=32 ... 
```

### Tuning number of model instances (multi-instance setup) along with intra/inter ops for parallel sections
```shell
python launcher.py --ninstances=4 -- src/main.py model=bert-base-cased batch_size=1 sequence_length=32 ...
```

### Tuning allocation library 
```shell
export TCMALLOC_LIBRARY_PATH=</path/to/tcmalloc/libtcmalloc.so>
python launcher.py --enable_tcmalloc -- src/main.py model=bert-base-cased batch_size=1 sequence_length=32 ...
```
 
### Tuning OpenMP implementation
```shell
export INTEL_OPENMP_LIBRARY_PATH=</path/to/intel/openmp/libomp.so>
python launcher.py --enable_iomp -- src/main.py model=bert-base-cased batch_size=1 sequence_length=32 ...
```

### Enabling Transparent Huge Page
```shell
python launcher.py --enable_thp -- src/main.py model=bert-base-cased batch_size=1 sequence_length=32 ...
```

## Hydra FAQ

## Executing dry-run to highlight configuration
```shell
python launcher.py --enable_tcmalloc --enable_iomp --ninstances=2 -- src/main.py --info config model=bert-base-cased batch_size=16 sequence_length=512
```
