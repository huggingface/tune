## HF Tune Docker Build Instructions

## Setup Github repo:
```
git clone https://github.com/karkadad/tune.git tune-ov
export TUNE_OV_DIR=`pwd`
cd tune-ov
git checkout ov-dev
```

### Step 1: Build Docker Container
Setup Docker name. Use this name, so our development can be consistent.
```
export HFTUNE_DOCKER=iotgedge/hftune-ov
```

#### OPTION 1
To build Docker container for HF Tune with OpenVINO backend:
```
cd $TUNE_OV_DIR/tune-ov/docker
docker build -t $HFTUNE_DOCKER -f Dockerfile.ov .././
```

#### OPTION 2
Pull the docker container. Docker container is [pushed here](https://hub.docker.com/r/iotgedge/hftune-ov)
```
docker pull $HFTUNE_DOCKER

```


### Step 2: Run the container

#### OPTION 1: With results persistant
In this option, we will mount our git cloned directory, so the results will be saved in our local directory. Use this option when benchmarking.

```
docker run --privileged --rm -v $TUNE_OV_DIR/tune-ov:/opt/tune -it $HFTUNE_DOCKER bash
```

#### OPTION 2: With results non-persistant. Use this for TESTING.
In this options, the results will be lost once we kill the docker container. So, use this for testing purposes.

```
docker run --privileged --rm -it $HFTUNE_DOCKER bash
```

**NOTE:** `--privileged` flag is needed when you use the launcher script. See [sample below](#launcher-script-sample).

### Step 3: Run the benchmarks

#### OPTION 1: Run by loggin into the container
```
# After running STEP 2, Now you will be inside the container.
root@0a6d974d2ebd:/opt/tune#
# Run sample Test
python3 src/main.py model=bert-base-cased sequence_length=32 backend=ov device=cpu
```

#### OPTION 2: Run directly by passing the benchmark cmd

```
docker run --privileged --rm -it $HFTUNE_DOCKER python3 src/main.py model=bert-base-cased sequence_length=32 backend=ov device=cpu
```

### Other working models:

```
python3 src/main.py model=bert-base-cased sequence_length=32 backend=ov device=cpu
python3 src/main.py model=gpt2 sequence_length=32 backend=ov device=cpu
python3 src/main.py model=roberta-base sequence_length=32 backend=ov device=cpu
python3 src/main.py model=distilbert-base-uncased sequence_length=32 backend=ov device=cpu
```
## Launcher Script Sample:
```
PYTHONPATH=src python3 launcher.py --multi_instance --ninstances=1 \
--kmp_affinity=verbose,granularity=fine,compact,1,0 --enable_iomp \
--enable_tcmalloc -- src/main.py --multirun backend=ort,ov \
batch_size=1 sequence_length=20 benchmark_duration=20 \
warmup_runs=1 model=distilbert-base-uncased
```