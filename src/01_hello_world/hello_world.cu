#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Simple CUDA kernel that prints from GPU threads
 */
__global__ void helloFromGPU() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU! Thread %d in Block %d (Global ID: %d)\n", 
           threadIdx.x, blockIdx.x, tid);
}

int main() {
    printf("=== CUDA Hello World ===\n");
    printf("Hello from CPU!\n\n");
    
    // Query GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
    
    printf("\n=== Launching Kernel ===\n");
    
    // Launch kernel with 2 blocks of 4 threads each
    int numBlocks = 2;
    int threadsPerBlock = 4;
    
    printf("Configuration: <<<%d blocks, %d threads per block>>>\n", 
           numBlocks, threadsPerBlock);
    printf("Total threads: %d\n\n", numBlocks * threadsPerBlock);
    
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();
    
    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n=== Kernel Execution Complete ===\n");
    
    return 0;
}
