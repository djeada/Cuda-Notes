# Performance Optimization in CUDA

## Key Performance Metrics

### 1. Throughput
- Amount of work completed per unit time
- Measured in GFLOPS (billion floating-point operations per second)

### 2. Latency
- Time to complete a single operation
- GPUs trade latency for throughput

### 3. Bandwidth
- Rate of data transfer
- Memory bandwidth is often the bottleneck

### 4. Occupancy
- Ratio of active warps to maximum possible warps
- Higher occupancy can hide latency

## Optimization Strategies

### 1. Maximize Occupancy

Occupancy is the ratio of active warps to the maximum number of warps supported by an SM.

**Factors Affecting Occupancy:**
- Threads per block
- Register usage per thread
- Shared memory usage per block

**Finding Optimal Configuration:**
```cuda
// Query occupancy
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                   myKernel, 0, 0);

// Calculate occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
    myKernel, blockSize, 0);
```

**Example:**
```cuda
// Calculate grid size for full occupancy
int blockSize = 256;
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
    kernel, blockSize, 0);

int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);

int gridSize = numBlocks * prop.multiProcessorCount;
kernel<<<gridSize, blockSize>>>(...);
```

### 2. Optimize Memory Access

#### Coalescing Global Memory Access

**Good Pattern:**
```cuda
__global__ void coalesced(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = tid; // Sequential access
    }
}
```

**Bad Pattern:**
```cuda
__global__ void strided(float *data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid * stride] = tid; // Non-coalesced
    }
}
```

#### Use Shared Memory for Reused Data

```cuda
__global__ void useShared(float *input, float *output, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load into shared memory
    if (gid < n) {
        cache[tid] = input[gid];
    }
    __syncthreads();
    
    // Multiple accesses to shared memory (fast)
    if (gid < n) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += cache[i];
        }
        output[gid] = sum;
    }
}
```

### 3. Minimize Warp Divergence

Warp divergence occurs when threads in a warp take different execution paths.

**Bad (Divergent):**
```cuda
__global__ void divergent(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (tid % 2 == 0) {
            // Even threads do this
            data[tid] *= 2;
        } else {
            // Odd threads do this - causes divergence
            data[tid] *= 3;
        }
    }
}
```

**Better (Less Divergence):**
```cuda
__global__ void lessDivergent(int *even, int *odd, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n/2) {
        even[tid] *= 2;
        odd[tid] *= 3;
    }
}
```

### 4. Reduce Register Pressure

High register usage reduces occupancy.

**Control Register Usage:**
```cuda
// Use compiler flag
nvcc -maxrregcount=32 kernel.cu

// Or use launch bounds
__global__ void __launch_bounds__(256, 4) kernel() {
    // Hints: max 256 threads/block, 4 blocks per SM
}
```

### 5. Loop Unrolling

**Manual Unrolling:**
```cuda
// Instead of
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// Unroll
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];
```

**Compiler Pragmas:**
```cuda
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}
```

### 6. Use Appropriate Math Functions

**Intrinsic Functions (Fast but Less Precise):**
```cuda
__device__ float fastIntrinsic(float x) {
    return __sinf(x);  // Fast sine
    // vs
    return sinf(x);    // Standard sine
}
```

**Common Intrinsics:**
- `__sinf()`, `__cosf()`, `__tanf()`
- `__expf()`, `__logf()`
- `__powf()`
- `__fdividef()` (fast division)

### 7. Streams and Concurrency

**Overlapping Computation and Communication:**
```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Async operations on different streams
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_a);

cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_b);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**Multiple Concurrent Kernels:**
```cuda
for (int i = 0; i < nStreams; i++) {
    int offset = i * streamSize;
    cudaMemcpyAsync(&d_data[offset], &h_data[offset], 
                    streamBytes, cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(&d_data[offset]);
    cudaMemcpyAsync(&h_data[offset], &d_data[offset], 
                    streamBytes, cudaMemcpyDeviceToHost, streams[i]);
}
```

### 8. Instruction-Level Parallelism (ILP)

Process multiple independent operations per thread.

```cuda
__global__ void ilp(float *out, const float *a, const float *b, int n) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (tid + 3 < n) {
        // Process 4 elements per thread (independent operations)
        float a0 = a[tid + 0], a1 = a[tid + 1];
        float a2 = a[tid + 2], a3 = a[tid + 3];
        float b0 = b[tid + 0], b1 = b[tid + 1];
        float b2 = b[tid + 2], b3 = b[tid + 3];
        
        out[tid + 0] = a0 + b0;
        out[tid + 1] = a1 + b1;
        out[tid + 2] = a2 + b2;
        out[tid + 3] = a3 + b3;
    }
}
```

## Profiling and Analysis

### Using nvprof (Legacy)

```bash
# Basic profiling
nvprof ./program

# Detailed metrics
nvprof --metrics achieved_occupancy,gld_efficiency ./program

# Timeline
nvprof --print-gpu-trace ./program
```

### Using Nsight Compute

```bash
# Profile kernel
ncu --set full ./program

# Specific metrics
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./program

# Generate report
ncu -o profile_report ./program
```

### Using Nsight Systems

```bash
# System-wide profiling
nsys profile --stats=true ./program

# Generate report
nsys profile -o report ./program
```

### Key Metrics to Monitor

1. **Achieved Occupancy** - Target: >50%
2. **Memory Throughput** - Compare to peak bandwidth
3. **SM Efficiency** - Percentage of time SMs are active
4. **Warp Execution Efficiency** - Impact of divergence
5. **Global Load/Store Efficiency** - Coalescing effectiveness

## Example: Optimized Reduction

```cuda
__global__ void reduce(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first level of reduction in global memory
    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockDim.x >= 64) smem[tid] = mySum = mySum + smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] = mySum = mySum + smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] = mySum = mySum + smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] = mySum = mySum + smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] = mySum = mySum + smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] = mySum = mySum + smem[tid + 1];
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

## Common Performance Pitfalls

### 1. Host-Device Transfer Overhead
**Problem:** Too many small transfers
**Solution:** Batch transfers, use pinned memory, overlap with computation

### 2. Uncoalesced Memory Access
**Problem:** Inefficient memory bandwidth usage
**Solution:** Restructure data layout, use AoS vs SoA appropriately

### 3. Low Occupancy
**Problem:** Not enough threads to hide latency
**Solution:** Reduce register usage, reduce shared memory usage, adjust block size

### 4. Bank Conflicts
**Problem:** Serialized access to shared memory
**Solution:** Pad arrays, change access patterns

### 5. Warp Divergence
**Problem:** Reduced parallel efficiency
**Solution:** Reorganize data, separate divergent paths

## Optimization Workflow

1. **Profile** - Identify bottlenecks
2. **Analyze** - Understand the cause
3. **Optimize** - Apply appropriate technique
4. **Verify** - Measure improvement
5. **Iterate** - Repeat for next bottleneck

## Best Practices Checklist

- [ ] Maximize parallel execution
- [ ] Optimize memory usage
- [ ] Minimize data transfers
- [ ] Coalesce global memory accesses
- [ ] Use shared memory effectively
- [ ] Avoid warp divergence
- [ ] Hide latency with sufficient threads
- [ ] Use appropriate precision
- [ ] Profile and measure
- [ ] Test on target hardware

## Resources

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Optimization Examples](https://github.com/NVIDIA/cuda-samples)
