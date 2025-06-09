#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>

// Define block size for CUDA kernel
#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    // Calculate row and column indices for the output matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0.0f;

    // Loop over all the sub-matrices of A and B that are required to
    // compute the block sub-matrix
    for (int tile = 0; tile < (N / BLOCK_SIZE); ++tile) {

        // Load sub-matrices of A and B into shared memory
        // Coalesced memory access
        As[threadIdx.y][threadIdx.x] = A[row * N + tile * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];

        __syncthreads(); // Ensure all threads have loaded their data

        // Perform the computation for this sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // Ensure all threads have completed the computation
    }

    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// CPU-based matrix multiplication for verification
void matrixMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// Unit tests
void testMatrixMultiplication(int N) {
    // Allocate host memory
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);
    std::vector<float> h_C_CPU(N * N);

    // Initialize matrices A and B with sample data
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N * N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Launch the CUDA kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy the result from device to host
    cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result using CPU-based matrix multiplication
    matrixMulCPU(h_A.data(), h_B.data(), h_C_CPU.data(), N);

    // Compare the results
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (abs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            std::cerr << "Error at element " << i << ": " << h_C[i] << " != " << h_C_CPU[i] << std::endl;
            correct = false;
            break;
        }
    }

    assert(correct);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Test with a small matrix size
    testMatrixMultiplication(32);

    // Test with a medium matrix size
    testMatrixMultiplication(64);

    // Test with a larger matrix size
    testMatrixMultiplication(128);

    std::cout << "All matrix multiplication tests passed!" << std::endl;

    return 0;
}
