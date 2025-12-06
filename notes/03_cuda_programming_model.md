# CUDA Programming Model

## Thread Hierarchy

CUDA organizes threads into a hierarchical structure that maps to the GPU's physical architecture.

### Grid, Blocks, and Threads

```
Grid
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   └── ...
├── Block (0,1)
│   └── ...
└── Block (1,0)
    └── ...
```

### Dimensions

Grids and blocks can be 1D, 2D, or 3D:

```cuda
// 1D configuration
dim3 blocks(256);
dim3 threads(512);

// 2D configuration
dim3 blocks(16, 16);
dim3 threads(32, 32);

// 3D configuration
dim3 blocks(8, 8, 8);
dim3 threads(8, 8, 8);
```

### Built-in Variables

Within a kernel, CUDA provides built-in variables to identify threads:

- `threadIdx.{x,y,z}`: Thread index within a block
- `blockIdx.{x,y,z}`: Block index within a grid
- `blockDim.{x,y,z}`: Dimensions of a block
- `gridDim.{x,y,z}`: Dimensions of a grid

### Computing Global Thread ID

```cuda
// 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int tid = y * (gridDim.x * blockDim.x) + x;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

## Kernel Launches

### Basic Syntax

```cuda
kernelName<<<gridSize, blockSize>>>(arguments);
```

### With Shared Memory and Stream

```cuda
kernelName<<<gridSize, blockSize, sharedMemSize, stream>>>(arguments);
```

### Example

```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1024;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
}
```

## Memory Management

### Memory Allocation

```cuda
// Allocate device memory
float *d_array;
cudaMalloc(&d_array, size * sizeof(float));

// Free device memory
cudaFree(d_array);

// Allocate host memory (pinned)
float *h_array;
cudaMallocHost(&h_array, size * sizeof(float));
cudaFreeHost(h_array);
```

### Memory Transfer

```cuda
// Host to Device
cudaMemcpy(d_array, h_array, size * sizeof(float), 
           cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_array, d_array, size * sizeof(float), 
           cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_dst, d_src, size * sizeof(float), 
           cudaMemcpyDeviceToDevice);

// Asynchronous transfer
cudaMemcpyAsync(d_array, h_array, size * sizeof(float), 
                cudaMemcpyHostToDevice, stream);
```

### Memory Initialization

```cuda
// Set memory to a value
cudaMemset(d_array, 0, size * sizeof(float));
```

## Function Qualifiers

### `__global__`
- Called from host, executed on device
- Must return `void`
- Entry point for kernel launches

```cuda
__global__ void myKernel() {
    // GPU code
}
```

### `__device__`
- Called from device, executed on device
- Can return any type
- Only callable from other device functions or kernels

```cuda
__device__ float square(float x) {
    return x * x;
}
```

### `__host__`
- Called from host, executed on host
- Can be combined with `__device__`

```cuda
__host__ __device__ float multiply(float a, float b) {
    return a * b;
}
```

## Variable Qualifiers

### `__shared__`
- Shared memory variable
- Shared among threads in a block
- Fast on-chip memory

```cuda
__shared__ float cache[256];
```

### `__constant__`
- Constant memory
- Read-only from kernels
- Cached for fast access

```cuda
__constant__ float params[64];
```

### `__device__`
- Global memory variable
- Accessible from all kernels
- Resides in device memory

```cuda
__device__ int counter;
```

## Synchronization

### Thread Synchronization

```cuda
__syncthreads(); // Synchronize threads in a block
```

### Device Synchronization

```cuda
cudaDeviceSynchronize(); // Wait for all kernels to complete
```

### Stream Synchronization

```cuda
cudaStreamSynchronize(stream); // Wait for specific stream
```

## Error Handling

```cuda
cudaError_t err = cudaMalloc(&d_array, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}

// Check last error
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
```

## Example: Complete Vector Addition

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1 << 20; // 1M elements
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

## Best Practices

1. **Choose appropriate grid/block dimensions**
   - Multiples of warp size (32)
   - Consider hardware limits
   - Balance occupancy

2. **Minimize memory transfers**
   - Keep data on GPU as long as possible
   - Use pinned memory for faster transfers

3. **Check for errors**
   - Always check return values
   - Use `cudaGetLastError()` after kernel launches

4. **Use asynchronous operations**
   - Overlap computation and communication
   - Use streams for concurrent execution

## Resources

- [CUDA Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
