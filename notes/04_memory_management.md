# Memory Management in CUDA

## Memory Hierarchy

Understanding CUDA's memory hierarchy is crucial for writing efficient GPU programs.

### Memory Types Overview

| Memory Type | Location | Cached | Access | Scope | Lifetime |
|-------------|----------|--------|--------|-------|----------|
| Register | On-chip | N/A | R/W | Thread | Thread |
| Local | Off-chip DRAM | Yes (L1/L2) | R/W | Thread | Thread |
| Shared | On-chip | N/A | R/W | Block | Block |
| Global | Off-chip DRAM | Yes (L2, L1 optional) | R/W | All threads | Application |
| Constant | Off-chip DRAM | Yes (constant cache) | R | All threads | Application |
| Texture | Off-chip DRAM | Yes (texture cache) | R | All threads | Application |

## Global Memory

### Allocation and Deallocation

```cuda
// Basic allocation
float *d_data;
cudaMalloc(&d_data, size * sizeof(float));
cudaFree(d_data);

// 2D allocation
float *d_matrix;
size_t pitch;
cudaMallocPitch(&d_matrix, &pitch, width * sizeof(float), height);
cudaFree(d_matrix);

// 3D allocation
cudaPitchedPtr d_volume;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaMalloc3D(&d_volume, extent);
cudaFree(d_volume.ptr);
```

### Memory Transfer

```cuda
// Synchronous transfer
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);

// Asynchronous transfer (requires pinned memory)
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// 2D memory transfer
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

// 3D memory transfer
cudaMemcpy3D(&params);
```

### Memory Coalescing

Global memory accesses are most efficient when threads in a warp access consecutive memory addresses.

**Good (Coalesced):**
```cuda
__global__ void coalesced(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid; // Sequential access
}
```

**Bad (Uncoalesced):**
```cuda
__global__ void uncoalesced(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * 32] = tid; // Strided access - inefficient
}
```

## Shared Memory

Shared memory is fast on-chip memory shared among threads in a block.

### Static Allocation

```cuda
__global__ void staticShared() {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    cache[tid] = tid;
    __syncthreads();
    // Use cache
}
```

### Dynamic Allocation

```cuda
__global__ void dynamicShared(int n) {
    extern __shared__ float cache[];
    int tid = threadIdx.x;
    cache[tid] = tid;
    __syncthreads();
    // Use cache
}

// Launch with shared memory size
kernel<<<grid, block, sharedMemSize>>>(n);
```

### Bank Conflicts

Shared memory is divided into 32 banks. Simultaneous access to the same bank (by different addresses) causes conflicts.

**No Conflict:**
```cuda
__shared__ float data[256];
float value = data[threadIdx.x]; // Each thread accesses different bank
```

**Bank Conflict:**
```cuda
__shared__ float data[256];
float value = data[threadIdx.x * 2]; // Stride causes conflicts
```

**Avoiding Conflicts with Padding:**
```cuda
__shared__ float data[256 + 8]; // Add padding to avoid conflicts
```

## Constant Memory

Constant memory is cached and optimized for broadcast (all threads reading the same address).

### Declaration and Initialization

```cuda
__constant__ float constData[256];

// In host code
float h_data[256];
// Initialize h_data...
cudaMemcpyToSymbol(constData, h_data, sizeof(float) * 256);
```

### Usage

```cuda
__global__ void useConstant() {
    float value = constData[10]; // Fast if all threads read same value
}
```

## Pinned (Page-Locked) Memory

Pinned memory enables faster transfers and asynchronous operations.

### Allocation

```cuda
float *h_pinned;
cudaMallocHost(&h_pinned, size * sizeof(float));
// Or
cudaHostAlloc(&h_pinned, size * sizeof(float), cudaHostAllocDefault);

cudaFreeHost(h_pinned);
```

### Benefits

1. Faster transfers (2x or more)
2. Enables asynchronous transfers
3. Enables mapped memory
4. Required for concurrent copy and execute

### Mapped Memory (Zero-Copy)

```cuda
float *h_mapped, *d_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);

// Kernel can directly access h_mapped through d_mapped
kernel<<<grid, block>>>(d_mapped);

cudaFreeHost(h_mapped);
```

## Unified Memory

Unified Memory creates a managed memory pool accessible from both CPU and GPU.

### Allocation

```cuda
float *data;
cudaMallocManaged(&data, size * sizeof(float));

// Use on CPU
for (int i = 0; i < n; i++) {
    data[i] = i;
}

// Use on GPU
kernel<<<grid, block>>>(data);
cudaDeviceSynchronize();

// Use on CPU again
printf("%f\n", data[0]);

cudaFree(data);
```

### Advantages

- Simplified memory management
- Automatic data migration
- Single pointer for CPU and GPU

### Considerations

- May have performance overhead
- Requires compute capability 3.0+
- Not always optimal for all workloads

## Memory Optimization Strategies

### 1. Maximize Memory Throughput

```cuda
// Use __restrict__ to help compiler optimize
__global__ void kernel(float * __restrict__ a, 
                       const float * __restrict__ b) {
    // Compiler knows a and b don't overlap
}
```

### 2. Minimize Transfers

```cuda
// Bad: Multiple small transfers
cudaMemcpy(d_a, h_a, size1, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size2, cudaMemcpyHostToDevice);

// Good: Single large transfer
struct Data { float *a; float *b; };
cudaMemcpy(d_data, h_data, totalSize, cudaMemcpyHostToDevice);
```

### 3. Use Shared Memory for Reused Data

```cuda
__global__ void tiled_matrix_mul(float *A, float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Load tile into shared memory
    As[threadIdx.y][threadIdx.x] = A[...];
    Bs[threadIdx.y][threadIdx.x] = B[...];
    __syncthreads();
    
    // Compute using shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
}
```

### 4. Align Data Structures

```cuda
// Align to 256 bytes for optimal coalescing
__align__(256) struct MyData {
    float values[64];
};
```

## Memory Access Patterns

### Sequential Access (Best)
```cuda
data[tid] = value; // tid = 0, 1, 2, 3, ...
```

### Strided Access (Poor)
```cuda
data[tid * stride] = value; // Large gaps between accesses
```

### Random Access (Worst)
```cuda
data[random_index[tid]] = value; // Completely random
```

## Debugging Memory Issues

### CUDA Memory Checker

```bash
cuda-memcheck ./program
```

### Common Issues

1. **Out of bounds access**
2. **Race conditions**
3. **Memory leaks**
4. **Uninitialized memory**
5. **Bank conflicts**

### Checking for Errors

```cuda
cudaError_t err = cudaMemcpy(...);
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}

// Check last error
cudaDeviceSynchronize();
err = cudaGetLastError();
```

## Example: Optimized Matrix Transpose

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(float *out, const float *in, 
                                   int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile into shared memory (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = 
                in[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write transposed tile (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = 
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

## Best Practices

1. **Minimize host-device transfers** - Keep data on GPU
2. **Use shared memory** for frequently accessed data
3. **Coalesce global memory accesses** - Sequential access patterns
4. **Avoid bank conflicts** in shared memory
5. **Use pinned memory** for faster transfers
6. **Consider Unified Memory** for prototyping
7. **Profile memory usage** with nvprof or Nsight

## Resources

- [CUDA Memory Management Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
