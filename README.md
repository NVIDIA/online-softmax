# Benchmark

This is a benchmark to explore performance of Softmax and Softmax+TopK functions with online calculation of softmax normalization term. Please see ["Online normalizer calculation for softmax"](https://arxiv.org/abs/1805.02867) paper.

# Build

You will need [CUB](https://nvlabs.github.io/cub/) v1.8.0 (or newer) and CUDA 9.1 (or newer) to build the benchmark. Set CUB variable in Makefile to the correct location of CUB library. Build the sample:
    make

# Run
Run the sample:
    ./online_softmax_benchmark
