FROM ubuntu:20.04

ARG TRANSFORMERS_VERSION=4.1.1
ARG PYTORCH_VERSION=1.7.1
ARG TENSORFLOW_VERSION=2.4.0
ARG ONNXRUNTIME_VERSION=1.6.0
ARG MKL_THREADING_LIBRARY=OMP

RUN apt update && \
    apt install -y \
        git \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# PyTorch
RUN python3 -m pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# TensorFlow
RUN python3 -m pip install tensorflow

# ONNX Runtime
RUN python3 -m pip install onnxruntime

COPY . /opt/intel-benchmarks

WORKDIR /opt/intel-benchmarks
RUN python3 -m pip install -r requirements.txt

