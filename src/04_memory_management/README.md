# Memory Management CUDA Example

## Description

This example demonstrates various memory types and management techniques in CUDA, including:
- Global memory
- Shared memory (static and dynamic)
- Constant memory
- Unified (managed) memory
- Pinned (page-locked) memory

## Key Concepts

### Memory Types

1. **Global Memory**
   - Largest but slowest
   - Accessible by all threads
   - Main working memory

2. **Shared Memory**
   - Fast on-chip memory
   - Shared within a thread block
   - Requires explicit management
   - Two types: static and dynamic allocation

3. **Constant Memory**
   - Read-only from kernels
   - Cached for efficient broadcast
   - 64 KB size limit

4. **Unified Memory**
   - Single pointer for CPU and GPU
   - Automatic data migration
   - Simplifies memory management

5. **Pinned Memory**
   - Page-locked host memory
   - Faster CPU-GPU transfers
   - Limited system resource

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./memory_demo
```

## Expected Output

The program will:
1. Display device memory specifications
2. Demonstrate unified memory with automatic migration
3. Compare transfer speeds between pageable and pinned memory
4. Show constant memory usage for efficient broadcast
5. Demonstrate shared memory for intra-block communication

## Performance Insights

### Transfer Speed Comparison
- Pageable memory: ~6-8 GB/s (system dependent)
- Pinned memory: ~10-12 GB/s (typically 1.5-2x faster)

### Memory Hierarchy (fastest to slowest)
1. Registers (per-thread)
2. Shared Memory (per-block)
3. L1 Cache
4. L2 Cache  
5. Global Memory
6. Host Memory

## Best Practices

1. **Use shared memory** for frequently accessed data within a block
2. **Use pinned memory** for large or frequent host-device transfers
3. **Use constant memory** when all threads read the same data
4. **Use unified memory** for prototyping or irregular access patterns
5. **Minimize transfers** between host and device
6. **Coalesce global memory accesses** for maximum bandwidth

## Requirements

- CUDA Toolkit (version 11.0 or higher recommended)
- NVIDIA GPU with compute capability 5.0 or higher (3.0+ for unified memory)
- CMake 3.18 or higher
