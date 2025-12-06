# Matrix Multiplication CUDA Example

## Description

This example demonstrates parallel matrix multiplication on the GPU using shared memory tiling for optimization. It computes `C = A × B` where matrices have dimensions:
- A: M × K
- B: K × N  
- C: M × N

## Key Concepts

- 2D thread blocks and grids
- Shared memory for data reuse
- Tiling to reduce global memory access
- Thread synchronization with `__syncthreads()`
- GFLOPS performance measurement
- 2D indexing

## Implementation Details

### Naive Implementation
- Each thread computes one element of the output matrix
- Reads data directly from global memory
- Simple but inefficient due to repeated memory accesses

### Tiled Implementation (Optimized)
- Uses shared memory to cache tiles of matrices
- Reduces global memory accesses by factor of TILE_SIZE
- Achieves significantly better performance
- Typical speedup: 5-10x over naive version

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
# Default: 1024x1024 matrices
./matrix_mul

# Custom sizes: M K N
./matrix_mul 2048 2048 2048
```

## Expected Output

The program will:
1. Initialize random matrices A and B
2. Copy data to GPU
3. Execute tiled matrix multiplication
4. Measure execution time and GFLOPS
5. Verify results (for small matrices)
6. Display sample output

## Performance Notes

- Matrix multiplication is compute-bound
- Performance measured in GFLOPS (Giga Floating-Point Operations Per Second)
- Tiled implementation uses shared memory for better cache locality
- Typical performance: 100-1000+ GFLOPS depending on GPU
- Larger matrices generally achieve better performance

## Optimization Techniques

1. **Shared Memory Tiling**: Reduces global memory bandwidth
2. **Coalesced Access**: Threads access consecutive memory
3. **Thread Reuse**: Each thread computes multiple operations
4. **Minimal Synchronization**: Only sync when necessary

## Requirements

- CUDA Toolkit (version 11.0 or higher recommended)
- NVIDIA GPU with compute capability 5.0 or higher
- CMake 3.18 or higher
- Sufficient GPU memory for large matrices
