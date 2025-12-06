cmake_minimum_required(VERSION 3.18)
project(matrix_multiplication LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(matrix_mul matrix_mul.cu)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(matrix_mul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80"
)

# Link CUDA runtime
target_link_libraries(matrix_mul CUDA::cudart)
