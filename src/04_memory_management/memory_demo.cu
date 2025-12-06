#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

// Constant memory declaration
__constant__ float constData[ARRAY_SIZE];

/**
 * Kernel demonstrating global memory access
 */
__global__ void globalMemoryDemo(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

/**
 * Kernel demonstrating shared memory usage
 */
__global__ void sharedMemoryDemo(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    
    // Load data into shared memory
    if (tid < n) {
        sharedData[localIdx] = input[tid];
    } else {
        sharedData[localIdx] = 0.0f;
    }
    
    __syncthreads();
    
    // Use shared memory for computation
    float sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        sum += sharedData[i];
    }
    
    if (tid < n) {
        output[tid] = sum;
    }
}

/**
 * Kernel demonstrating constant memory access
 */
__global__ void constantMemoryDemo(float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // All threads read same constant value (efficient broadcast)
        float value = constData[0];
        output[tid] = value * tid;
    }
}

/**
 * Kernel using dynamic shared memory
 */
__global__ void dynamicSharedDemo(float *input, float *output, int n) {
    extern __shared__ float dynamicShared[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    
    if (tid < n) {
        dynamicShared[localIdx] = input[tid] * 2.0f;
    }
    __syncthreads();
    
    // Simple reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride && tid + stride < n) {
            dynamicShared[localIdx] += dynamicShared[localIdx + stride];
        }
        __syncthreads();
    }
    
    if (localIdx == 0 && blockIdx.x * blockDim.x < n) {
        output[blockIdx.x] = dynamicShared[0];
    }
}

/**
 * Demonstrate unified memory
 */
void unifiedMemoryDemo() {
    printf("\n=== Unified Memory Demo ===\n");
    
    int n = 1024;
    size_t bytes = n * sizeof(float);
    
    // Allocate unified memory
    float *data;
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    
    // Initialize on CPU
    printf("Initializing data on CPU...\n");
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }
    
    // Launch kernel (data automatically migrates to GPU)
    int blocks = (n + 255) / 256;
    globalMemoryDemo<<<blocks, 256>>>(data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Access on CPU (data automatically migrates back)
    printf("Sample results: %.1f, %.1f, %.1f\n", data[0], data[1], data[2]);
    
    CHECK_CUDA(cudaFree(data));
    printf("✓ Unified memory demo complete\n");
}

/**
 * Demonstrate pinned (page-locked) memory
 */
void pinnedMemoryDemo() {
    printf("\n=== Pinned Memory Demo ===\n");
    
    int n = 1 << 24; // 16M elements
    size_t bytes = n * sizeof(float);
    
    float *h_pageable, *h_pinned;
    float *d_data;
    
    // Allocate pageable host memory
    h_pageable = (float*)malloc(bytes);
    
    // Allocate pinned host memory
    CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_pageable[i] = i;
        h_pinned[i] = i;
    }
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float milliseconds;
    
    // Test pageable memory transfer
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Pageable transfer time: %.3f ms (%.2f GB/s)\n", 
           milliseconds, (bytes / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0));
    
    // Test pinned memory transfer
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Pinned transfer time: %.3f ms (%.2f GB/s)\n", 
           milliseconds, (bytes / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0));
    
    // Cleanup
    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("✓ Pinned memory demo complete\n");
}

/**
 * Demonstrate constant memory
 */
void constantMemoryDemo() {
    printf("\n=== Constant Memory Demo ===\n");
    
    int n = 1024;
    size_t bytes = n * sizeof(float);
    
    // Initialize constant memory from host
    float h_const[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_const[i] = i * 0.5f;
    }
    CHECK_CUDA(cudaMemcpyToSymbol(constData, h_const, sizeof(float) * ARRAY_SIZE));
    
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    int blocks = (n + 255) / 256;
    constantMemoryDemo<<<blocks, 256>>>(d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float h_output[10];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 10 * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    printf("Sample results from constant memory:\n");
    for (int i = 0; i < 5; i++) {
        printf("  output[%d] = %.2f\n", i, h_output[i]);
    }
    
    CHECK_CUDA(cudaFree(d_output));
    printf("✓ Constant memory demo complete\n");
}

/**
 * Main function demonstrating various memory types
 */
int main() {
    printf("=== CUDA Memory Management Demo ===\n");
    
    // Query device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n", prop.name);
    printf("Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %lu KB\n", 
           prop.sharedMemPerBlock / 1024);
    printf("Constant Memory: 64 KB\n");
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    
    // Run demos
    unifiedMemoryDemo();
    pinnedMemoryDemo();
    constantMemoryDemo();
    
    printf("\n=== All Memory Demos Complete ===\n");
    return EXIT_SUCCESS;
}
