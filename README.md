# CUDA Notes

A comprehensive collection of CUDA programming notes, examples, and resources for learning GPU parallel programming.

## Repository Structure

```
Cuda-Notes/
├── notes/          # Theoretical notes and concepts
│   ├── 01_introduction_to_cuda.md
│   ├── 02_gpu_architecture.md
│   ├── 03_cuda_programming_model.md
│   ├── 04_memory_management.md
│   └── 05_performance_optimization.md
│
└── src/            # Example code projects (each with own CMake)
    ├── 01_hello_world/
    ├── 02_vector_addition/
    ├── 03_matrix_multiplication/
    └── 04_memory_management/
```

## Getting Started

### Prerequisites

- NVIDIA GPU with compute capability 3.0 or higher
- CUDA Toolkit (11.0 or higher recommended)
- CMake 3.18 or higher
- C/C++ compiler (GCC, Clang, or MSVC)

### Installation

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your platform

2. **Verify Installation**
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Clone this repository**
   ```bash
   git clone https://github.com/djeada/Cuda-Notes.git
   cd Cuda-Notes
   ```

### Building Examples

Each example in the `src/` directory has its own CMakeLists.txt:

```bash
cd src/01_hello_world
mkdir build && cd build
cmake ..
make
./hello_world
```

## Notes

The `/notes` directory contains detailed theoretical documentation:

1. **[Introduction to CUDA](notes/01_introduction_to_cuda.md)** - CUDA basics, architecture, and workflow
2. **[GPU Architecture](notes/02_gpu_architecture.md)** - Understanding GPU hardware and design
3. **[CUDA Programming Model](notes/03_cuda_programming_model.md)** - Thread hierarchy, kernels, and memory management
4. **[Memory Management](notes/04_memory_management.md)** - Memory types, optimization, and best practices
5. **[Performance Optimization](notes/05_performance_optimization.md)** - Techniques for maximizing GPU performance

## Example Projects

### 1. Hello World
Basic CUDA program demonstrating kernel launches and thread identification.

**Key Concepts:** Kernel syntax, thread indexing, device queries

### 2. Vector Addition
Parallel element-wise vector addition with performance measurement.

**Key Concepts:** Memory allocation, data transfers, coalesced access, bandwidth calculation

### 3. Matrix Multiplication
Optimized matrix multiplication using shared memory tiling.

**Key Concepts:** 2D grids, shared memory, synchronization, GFLOPS measurement

### 4. Memory Management
Comprehensive demonstration of CUDA memory types and optimization techniques.

**Key Concepts:** Global, shared, constant, unified, and pinned memory

## Resources

### Official NVIDIA Documentation

#### CUDA Programming Guides
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - Comprehensive programming guide
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) - Performance optimization guidelines
- [CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) - Compiler documentation
- [CUDA NVCC Basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html) - NVCC compiler basics

#### API References
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) - Runtime API reference
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) - Lower-level driver API
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) - Linear algebra library
- [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) - Fast Fourier Transform library
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/api/index.html) - Deep neural network library
- [Thrust](https://docs.nvidia.com/cuda/thrust/index.html) - C++ template library for CUDA

#### Tools and Profiling
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/index.html) - Kernel profiler
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/index.html) - System-wide profiler
- [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html) - CUDA debugger
- [CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/index.html) - Memory error detector

#### GPU Architecture
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/) - Detailed architecture documentation
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/) - Ampere GPU architecture
- [Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) - Latest Hopper architecture

### Learning Resources

#### Online Courses
- [NVIDIA DLI CUDA Training](https://www.nvidia.com/en-us/training/) - Official NVIDIA training courses
- [Udacity: Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344) - Free course on GPU programming
- [Coursera: GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming) - Comprehensive GPU programming courses

#### Books
- **"Programming Massively Parallel Processors"** by David Kirk and Wen-mei Hwu - Comprehensive CUDA textbook
- **"CUDA by Example"** by Jason Sanders and Edward Kandrot - Practical introduction to CUDA
- **"Professional CUDA C Programming"** by John Cheng, Max Grossman, Ty McKercher - Advanced CUDA techniques
- **"CUDA Programming: A Developer's Guide to Parallel Computing"** by Shane Cook - Detailed programming guide

#### Video Tutorials
- [NVIDIA Developer YouTube Channel](https://www.youtube.com/c/NVIDIADeveloper) - Official tutorials and presentations
- [CUDA Crash Course](https://www.youtube.com/watch?v=2DYm3futg-U) - Quick introduction to CUDA
- [GPU Computing Tutorials](https://www.nvidia.com/en-us/on-demand/session-library/) - NVIDIA GTC presentations

### Community and Forums

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/206) - Official CUDA forums
- [Stack Overflow CUDA Tag](https://stackoverflow.com/questions/tagged/cuda) - Q&A community
- [Reddit r/CUDA](https://www.reddit.com/r/CUDA/) - CUDA community discussions
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog) - Latest news and techniques

### Code Repositories and Examples

- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - Official NVIDIA CUDA examples
- [GPU Computing SDK](https://developer.nvidia.com/gpu-computing-sdk) - Code samples and utilities
- [CUDA by Example Code](https://developer.nvidia.com/cuda-example) - Code from the "CUDA by Example" book
- [Modern GPU](https://github.com/moderngpu/moderngpu) - Patterns and utilities for CUDA

### Tools and Libraries

#### Mathematical Libraries
- [cuBLAS](https://developer.nvidia.com/cublas) - Dense linear algebra
- [cuSPARSE](https://developer.nvidia.com/cusparse) - Sparse linear algebra
- [cuFFT](https://developer.nvidia.com/cufft) - Fast Fourier transforms
- [cuRAND](https://developer.nvidia.com/curand) - Random number generation
- [cuSOLVER](https://developer.nvidia.com/cusolver) - Dense and sparse direct solvers

#### Deep Learning
- [cuDNN](https://developer.nvidia.com/cudnn) - Deep neural network primitives
- [TensorRT](https://developer.nvidia.com/tensorrt) - Inference optimization
- [NCCL](https://developer.nvidia.com/nccl) - Multi-GPU communication

#### Other Libraries
- [Thrust](https://github.com/NVIDIA/thrust) - C++ parallel algorithms
- [CUB](https://github.com/NVIDIA/cub) - Reusable CUDA primitives
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA templates for linear algebra
- [ArrayFire](https://arrayfire.com/) - High-level GPU library

### Performance Analysis

- [CUDA Profiler User Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Performance Optimization Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-optimization-strategies)

### Research Papers

- [GPU Computing Gems](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_pref01.html) - Collection of GPU programming techniques
- [CUDA Research Papers](https://scholar.google.com/scholar?q=cuda+gpu+programming) - Academic research on GPU computing

### Additional Resources

- [CUDA Zone](https://developer.nvidia.com/cuda-zone) - Central hub for CUDA resources
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/tag/cuda/) - Technical articles and tutorials
- [Parallel Forall Blog](https://developer.nvidia.com/blog/tag/parallel-forall/) - Advanced CUDA techniques
- [GTC On-Demand](https://www.nvidia.com/en-us/on-demand/) - Conference presentations and training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for CUDA Toolkit and comprehensive documentation
- CUDA programming community for examples and best practices
- Contributors and maintainers of CUDA educational resources

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note:** This repository is for educational purposes. Make sure your system meets the hardware and software requirements before attempting to run the examples.