# CUDA Theory Notes

This directory contains comprehensive theoretical notes on CUDA programming and GPU computing.

## Table of Contents

### 1. [Introduction to CUDA](01_introduction_to_cuda.md)
Learn the fundamentals of CUDA, including:
- What is CUDA and why use it
- CUDA architecture overview (Host, Device, Kernels, Threads)
- CUDA-enabled GPUs and compute capabilities
- CUDA Toolkit components
- Basic workflow and Hello World example
- Getting started with CUDA development

### 2. [GPU Architecture](02_gpu_architecture.md)
Understand the hardware architecture of modern GPUs:
- GPU vs CPU architecture comparison
- Streaming Multiprocessors (SMs) and CUDA cores
- Memory hierarchy (registers, shared, L1/L2 cache, global memory)
- Compute capability across generations (Kepler to Hopper)
- SIMT execution model
- Thread hierarchy (Grid → Blocks → Warps → Threads)
- Modern GPU features (Tensor Cores, RT Cores, MPS)
- Querying GPU properties

### 3. [CUDA Programming Model](03_cuda_programming_model.md)
Master the CUDA programming model:
- Thread hierarchy (grids, blocks, threads)
- Built-in variables (threadIdx, blockIdx, blockDim, gridDim)
- Kernel launches and configuration
- Memory management (allocation, transfer, initialization)
- Function qualifiers (`__global__`, `__device__`, `__host__`)
- Variable qualifiers (`__shared__`, `__constant__`, `__device__`)
- Synchronization primitives
- Error handling
- Complete examples with best practices

### 4. [Memory Management](04_memory_management.md)
Deep dive into CUDA memory types and optimization:
- Memory hierarchy and types overview
- Global memory (allocation, transfer, coalescing)
- Shared memory (static/dynamic allocation, bank conflicts)
- Constant memory (broadcast optimization)
- Pinned (page-locked) memory for faster transfers
- Unified memory and automatic migration
- Memory optimization strategies
- Access patterns and performance implications
- Debugging memory issues
- Optimized examples (matrix transpose)

### 5. [Performance Optimization](05_performance_optimization.md)
Techniques for maximizing GPU performance:
- Key performance metrics (throughput, latency, bandwidth, occupancy)
- Maximizing occupancy
- Memory access optimization and coalescing
- Minimizing warp divergence
- Reducing register pressure
- Loop unrolling
- Fast math intrinsics
- Streams and concurrency
- Instruction-level parallelism (ILP)
- Profiling with nvprof, Nsight Compute, and Nsight Systems
- Key metrics to monitor
- Optimized reduction example
- Common performance pitfalls
- Optimization workflow and best practices

## How to Use These Notes

1. **Sequential Learning**: Start with Introduction and work through in order
2. **Reference Material**: Jump to specific topics as needed
3. **Companion to Examples**: Use alongside the code examples in `/src`
4. **Best Practices**: Each note includes best practices and gotchas

## Prerequisites

Basic knowledge of:
- C/C++ programming
- Computer architecture concepts
- Basic parallel programming concepts (helpful but not required)

## Additional Resources

For hands-on practice, see the example projects in the `/src` directory:
- `01_hello_world` - Basic kernel launches
- `02_vector_addition` - Memory management and data transfers
- `03_matrix_multiplication` - Shared memory and tiling
- `04_memory_management` - Comprehensive memory type examples

## Contributing

Found an error or want to add more content? Contributions are welcome! Please see the main repository README for contribution guidelines.

## Further Reading

For more in-depth information, refer to:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Main repository [README](../README.md) for extensive resource links
