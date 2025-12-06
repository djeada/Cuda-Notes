# Vector Addition CUDA Example

## Description

This example demonstrates parallel vector addition on the GPU. It adds two vectors element-wise: `C[i] = A[i] + B[i]`.

## Key Concepts

- Memory allocation with `cudaMalloc()`
- Data transfer with `cudaMemcpy()`
- Kernel launch configuration calculation
- Thread indexing and bounds checking
- Performance timing with CUDA events
- Bandwidth calculation
- Result verification

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
# Default: 1M elements
./vector_add

# Custom size (e.g., 10M elements)
./vector_add 10000000
```

## Expected Output

The program will:
1. Initialize two random vectors
2. Copy data to GPU memory
3. Launch the kernel with optimal block configuration
4. Measure execution time and bandwidth
5. Copy results back to host
6. Verify correctness
7. Display sample results

## Performance Notes

- Vector addition is memory-bound (limited by memory bandwidth)
- Typical bandwidth: 200-800 GB/s depending on GPU
- For optimal performance, use large vector sizes (>1M elements)
- Coalesced memory access pattern ensures maximum throughput

## Requirements

- CUDA Toolkit (version 11.0 or higher recommended)
- NVIDIA GPU with compute capability 5.0 or higher
- CMake 3.18 or higher
