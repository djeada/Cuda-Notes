#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

/**
 * CUDA kernel for vector addition
 * C[i] = A[i] + B[i]
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

/**
 * Initialize vector with values
 */
void initVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;
    }
}

/**
 * Verify results on CPU
 */
bool verifyResults(const float *A, const float *B, const float *C, int N) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > epsilon) {
            printf("Verification failed at index %d: %f + %f = %f (expected %f)\n",
                   i, A[i], B[i], C[i], expected);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // Vector size
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 20); // Default 1M elements
    size_t bytes = N * sizeof(float);
    
    printf("=== Vector Addition ===\n");
    printf("Vector size: %d elements (%.2f MB)\n", N, bytes / (1024.0 * 1024.0));
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize vectors
    printf("Initializing vectors...\n");
    initVector(h_A, N);
    initVector(h_B, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));
    
    // Copy data to device
    printf("Copying data to device...\n");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel: <<<%d blocks, %d threads>>>\n", 
           blocksPerGrid, threadsPerBlock);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Record start event
    CHECK_CUDA(cudaEventRecord(start));
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Check for kernel errors
    CHECK_CUDA(cudaGetLastError());
    
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Calculate bandwidth
    float bandwidth = (3.0f * bytes) / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    // Copy result back to host
    printf("Copying results back to host...\n");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    printf("Verifying results...\n");
    if (verifyResults(h_A, h_B, h_C, N)) {
        printf("✓ Results verified successfully!\n");
    } else {
        printf("✗ Verification failed!\n");
    }
    
    // Print sample results
    printf("\nSample results (first 10 elements):\n");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("  %.2f + %.2f = %.2f\n", h_A[i], h_B[i], h_C[i]);
    }
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\n=== Complete ===\n");
    return EXIT_SUCCESS;
}
