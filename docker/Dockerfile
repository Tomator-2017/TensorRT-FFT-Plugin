FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive

# The NGC PyTorch image already contains CUDA, cuFFT, TensorRT 8.6, PyTorch,
# ONNX and the required Python CUDA bindings. Install only build utilities.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mnt
