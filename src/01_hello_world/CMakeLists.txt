cmake_minimum_required(VERSION 3.18)
project(hello_world LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(hello_world hello_world.cu)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(hello_world PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80"
)

# Link CUDA runtime
target_link_libraries(hello_world CUDA::cudart)
