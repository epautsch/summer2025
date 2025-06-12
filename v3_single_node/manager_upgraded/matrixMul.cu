#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

int main() {
  int n = 2; // Matrix size
  size_t matrixSize = n * n * sizeof(float);

  // Host memory allocation
  float *h_A = new float[n * n];
  float *h_B = new float[n * n];
  float *h_C = new float[n * n];

  // Initialize matrices A and B (example values)
  for (int i = 0; i < n * n; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // Device memory allocation
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, matrixSize);
  cudaMalloc((void **)&d_B, matrixSize);
  cudaMalloc((void **)&d_C, matrixSize);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(2, 2);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

  // Launch the kernel
  matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

  // Copy data from device to host
  cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

  // Print the result matrix C
  std::cout << "Matrix C:\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << h_C[i * n + j] << " ";
    }
    std::cout << std::endl;
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
