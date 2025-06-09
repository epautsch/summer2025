#include <iostream>
#include <cuda_runtime.h>

// Define the matrix dimensions
#define M 1024
#define N 1024
#define P 1024

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int m, int n, int p) {
    // Calculate the row and column indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the output matrix
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    // Declare host matrices
    float *h_A = new float[M * N];
    float *h_B = new float[N * P];
    float *h_C = new float[M * P];

    // Initialize host matrices (example initialization)
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = (float)i;
    }
    for (int i = 0; i < N * P; ++i) {
        h_B[i] = (float)i;
    }

    // Declare device matrices
    float *d_A;
    float *d_B;
    float *d_C;

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * P * sizeof(float));
    cudaMalloc((void **)&d_C, M * P * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int blockSize = 16; // Adjust based on your GPU's capabilities
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, P);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy data from device to host
    cudaMemcpy(h_C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the results (optional)
    // ...

    // Free memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on the host
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
