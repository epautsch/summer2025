#include <iostream>
#include <vector>
#include <omp.h>

int main() {
  const int N = 256;
  std::vector<int> A(N * N, 1);
  std::vector<int> B(N * N, 2);
  std::vector<int> C(N * N, 0);

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }

  std::cout << "C[0][0] = " << C[0] << std::endl;

  return 0;
}
