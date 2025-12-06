# Hello World CUDA Example

## Description

This is a simple "Hello World" program that demonstrates the basic structure of a CUDA application. It prints messages from both the CPU (host) and GPU (device).

## Key Concepts

- `__global__` keyword for kernel functions
- Kernel launch syntax: `kernel<<<numBlocks, threadsPerBlock>>>()`
- `cudaDeviceSynchronize()` to wait for GPU completion
- Querying GPU properties with `cudaGetDeviceProperties()`
- Thread indexing with `blockIdx`, `blockDim`, and `threadIdx`

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./hello_world
```

## Expected Output

The program will:
1. Print a greeting from the CPU
2. Display information about available CUDA devices
3. Launch a kernel with 2 blocks of 4 threads each
4. Print greetings from all 8 GPU threads

## Requirements

- CUDA Toolkit (version 11.0 or higher recommended)
- NVIDIA GPU with compute capability 5.0 or higher
- CMake 3.18 or higher
