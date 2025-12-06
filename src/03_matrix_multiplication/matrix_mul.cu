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

#define TILE_SIZE 16

/**
 * Naive matrix multiplication kernel
 * C = A * B
 */
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Tiled matrix multiplication kernel using shared memory
 * C = A * B
 */
__global__ void matrixMulTiled(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Initialize matrix with random values
 */
void initMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

/**
 * CPU matrix multiplication for verification
 */
void matrixMulCPU(const float *A, const float *B, float *C,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Verify GPU results against CPU
 */
bool verifyResults(const float *gpu, const float *cpu, int size) {
    const float epsilon = 1e-2f;
    for (int i = 0; i < size; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", 
                   i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;
    
    printf("=== Matrix Multiplication ===\n");
    printf("Matrix sizes: C(%d x %d) = A(%d x %d) * B(%d x %d)\n",
           M, N, M, K, K, N);
    
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    
    printf("Memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n",
           bytesA / (1024.0 * 1024.0),
           bytesB / (1024.0 * 1024.0),
           bytesC / (1024.0 * 1024.0));
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    float *h_C_ref = (float*)malloc(bytesC);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Tiled version
    printf("\n--- Tiled Implementation ---\n");
    CHECK_CUDA(cudaEventRecord(start));
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Calculate GFLOPS
    float gflops = (2.0f * M * N * K) / (milliseconds / 1000.0f) / 1e9f;
    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    
    // Verify with small CPU computation
    if (M <= 512 && N <= 512 && K <= 512) {
        printf("\nVerifying results (CPU computation)...\n");
        matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);
        
        if (verifyResults(h_C, h_C_ref, M * N)) {
            printf("✓ Results verified successfully!\n");
        } else {
            printf("✗ Verification failed!\n");
        }
    } else {
        printf("\nSkipping verification (matrix too large)\n");
    }
    
    // Print sample results
    printf("\nSample result (top-left 3x3 of matrix C):\n");
    for (int i = 0; i < 3 && i < M; i++) {
        for (int j = 0; j < 3 && j < N; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
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
    free(h_C_ref);
    
    printf("\n=== Complete ===\n");
    return EXIT_SUCCESS;
}
