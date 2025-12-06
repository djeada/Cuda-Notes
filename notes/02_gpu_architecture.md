# GPU Architecture

## Overview

Understanding GPU architecture is crucial for writing efficient CUDA programs. Modern NVIDIA GPUs are designed for massive parallel processing.

## GPU vs CPU Architecture

### CPU Characteristics
- Few cores (4-64 typically)
- Complex control logic
- Large cache memory
- Optimized for sequential processing
- Low latency

### GPU Characteristics
- Thousands of cores
- Simple control logic per core
- Smaller cache per core
- Optimized for throughput
- High latency tolerance

## NVIDIA GPU Architecture

### Streaming Multiprocessors (SMs)

The GPU consists of multiple Streaming Multiprocessors (SMs), each containing:
- **CUDA Cores**: Perform integer and floating-point operations
- **Special Function Units (SFUs)**: Handle transcendental functions
- **Load/Store Units (LD/ST)**: Memory operations
- **Warp Schedulers**: Schedule threads for execution
- **Register File**: Fast on-chip memory for threads
- **Shared Memory**: Fast memory shared among threads in a block

### Memory Hierarchy

From fastest to slowest:

1. **Registers** (per-thread)
   - Fastest memory
   - Limited capacity
   - Private to each thread

2. **Shared Memory** (per-block)
   - Very fast on-chip memory
   - Shared among threads in the same block
   - Requires explicit management

3. **L1 Cache** (per-SM)
   - Automatic caching
   - Limited size

4. **L2 Cache** (per-GPU)
   - Shared across all SMs
   - Larger than L1

5. **Global Memory** (device DRAM)
   - Largest capacity
   - Highest latency
   - Accessible by all threads

6. **Constant Memory**
   - Cached and optimized for broadcast
   - Read-only from kernels

7. **Texture Memory**
   - Optimized for 2D spatial locality
   - Read-only from kernels

## Compute Capability

Different GPU generations have different compute capabilities:

- **3.x**: Kepler architecture
- **5.x**: Maxwell architecture
- **6.x**: Pascal architecture
- **7.x**: Volta, Turing architectures
- **8.x**: Ampere architecture
- **9.x**: Hopper architecture

Each generation introduces:
- More CUDA cores
- Better memory bandwidth
- New instructions
- Enhanced features

## Execution Model

### SIMT (Single Instruction, Multiple Thread)

- Threads are grouped into **warps** (32 threads)
- All threads in a warp execute the same instruction
- Warps execute independently
- Warp divergence can reduce performance

### Thread Hierarchy

```
Grid (entire kernel launch)
  └── Blocks (thread blocks)
      └── Warps (groups of 32 threads)
          └── Threads (individual execution units)
```

## Key Performance Considerations

1. **Memory Coalescing**: Access contiguous memory addresses
2. **Occupancy**: Keep SMs busy with enough threads
3. **Warp Divergence**: Avoid conditional branches that cause threads to diverge
4. **Bank Conflicts**: Ensure proper shared memory access patterns
5. **Memory Bandwidth**: Optimize data transfers

## Modern GPU Features

### Tensor Cores (Volta+)
- Specialized for matrix operations
- Accelerate deep learning workloads
- Mixed-precision computing

### RT Cores (Turing+)
- Ray tracing acceleration
- Real-time rendering

### Multi-Process Service (MPS)
- Share GPU among multiple processes
- Improved utilization

## Example: Querying GPU Properties

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", 
               prop.major, prop.minor);
        printf("  Total Global Memory: %lu MB\n", 
               prop.totalGlobalMem / (1024*1024));
        printf("  Multiprocessors: %d\n", 
               prop.multiProcessorCount);
        printf("  CUDA Cores: ~%d\n", 
               prop.multiProcessorCount * 128);
        printf("  Max Threads per Block: %d\n", 
               prop.maxThreadsPerBlock);
        printf("  Shared Memory per Block: %lu KB\n", 
               prop.sharedMemPerBlock / 1024);
    }
    
    return 0;
}
```

## Resources

- [CUDA Architecture Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
