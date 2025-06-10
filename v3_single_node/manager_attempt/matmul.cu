#include <iostream>
#include <cuda_runtime.h>

// Define the size of the matrices.  Adjust as needed.
#define N 1024

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    // Calculate the row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the matrix
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    // Host memory allocation
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N * N - i);
    }

    // Device memory allocation
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size as needed for optimal performance
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result (optional)
    // You can compare h_C with the expected result here.

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Matrix multiplication completed successfully." << std::endl;

    return 0;
}
