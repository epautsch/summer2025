cmake_minimum_required(VERSION 3.10)
project(matrixMul)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)

# Add the CUDA executable
add_executable(matrixMul matrixMul.cu)

# Link the CUDA executable with the CUDA libraries
target_link_libraries(matrixMul CUDA::cuda_runtime)

# Optional: Add include directories if needed
# target_include_directories(matrixMul PUBLIC ${CUDA_INCLUDE_DIRS})
