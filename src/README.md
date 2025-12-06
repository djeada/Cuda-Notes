# CUDA Example Projects

This directory contains practical CUDA programming examples. Each example is a standalone project with its own CMakeLists.txt for building.

## Projects Overview

### 01. [Hello World](01_hello_world/)
**Difficulty:** Beginner  
**Concepts:** Basic kernel syntax, thread identification, device queries

A simple introduction to CUDA programming that demonstrates:
- Writing and launching a CUDA kernel
- Understanding thread and block organization
- Querying GPU properties and capabilities
- Basic synchronization

**Files:**
- `hello_world.cu` - Main program with kernel
- `CMakeLists.txt` - Build configuration
- `README.md` - Detailed documentation

---

### 02. [Vector Addition](02_vector_addition/)
**Difficulty:** Beginner  
**Concepts:** Memory allocation, data transfers, parallel computation

Parallel element-wise vector addition (`C[i] = A[i] + B[i]`) demonstrating:
- CUDA memory allocation with `cudaMalloc()`
- Host-device data transfers with `cudaMemcpy()`
- Optimal kernel launch configuration
- Performance timing with CUDA events
- Bandwidth calculation
- Result verification

**Files:**
- `vector_add.cu` - Complete vector addition implementation
- `CMakeLists.txt` - Build configuration
- `README.md` - Usage and performance notes

---

### 03. [Matrix Multiplication](03_matrix_multiplication/)
**Difficulty:** Intermediate  
**Concepts:** 2D grids, shared memory, tiling optimization

Optimized matrix multiplication (`C = A × B`) featuring:
- 2D thread blocks and grids
- Shared memory tiling for performance
- Thread synchronization with `__syncthreads()`
- GFLOPS performance measurement
- Comparison of naive vs. optimized implementations

**Files:**
- `matrix_mul.cu` - Tiled matrix multiplication
- `CMakeLists.txt` - Build configuration
- `README.md` - Algorithm explanation and performance analysis

---

### 04. [Memory Management](04_memory_management/)
**Difficulty:** Intermediate  
**Concepts:** Memory types, optimization techniques

Comprehensive demonstration of CUDA memory types:
- Global memory access patterns
- Shared memory (static and dynamic)
- Constant memory for broadcast operations
- Unified (managed) memory
- Pinned (page-locked) memory for faster transfers
- Performance comparison of different memory types

**Files:**
- `memory_demo.cu` - Multiple memory type demonstrations
- `CMakeLists.txt` - Build configuration
- `README.md` - Memory hierarchy and best practices

---

## Building the Examples

### Prerequisites

- NVIDIA GPU with compute capability 3.0+ (5.0+ recommended)
- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- C++ compiler (GCC, Clang, or MSVC)

### Build Instructions

Each project has its own build directory. To build an example:

```bash
# Navigate to the project directory
cd 01_hello_world

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make

# Run the executable
./hello_world
```

### Build All Examples

To build all examples at once, you can use a script:

```bash
#!/bin/bash
for dir in 01_hello_world 02_vector_addition 03_matrix_multiplication 04_memory_management; do
    echo "Building $dir..."
    cd $dir
    mkdir -p build
    cd build
    cmake .. && make
    cd ../..
done
```

### Adjusting CUDA Architectures

The CMakeLists.txt files specify CUDA architectures as `"50;60;70;75;80"`. Adjust these based on your GPU:

| GPU Series | Compute Capability | Architecture Name |
|------------|-------------------|-------------------|
| GTX 900 series | 5.0, 5.2 | Maxwell |
| GTX 1000 series | 6.0, 6.1 | Pascal |
| RTX 2000 series | 7.5 | Turing |
| RTX 3000 series | 8.6 | Ampere |
| RTX 4000 series | 8.9 | Ada Lovelace |
| H100 | 9.0 | Hopper |

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Running the Examples

### Hello World
```bash
cd 01_hello_world/build
./hello_world
```

### Vector Addition
```bash
cd 02_vector_addition/build
./vector_add              # Default: 1M elements
./vector_add 10000000     # Custom size: 10M elements
```

### Matrix Multiplication
```bash
cd 03_matrix_multiplication/build
./matrix_mul              # Default: 1024x1024
./matrix_mul 2048 2048 2048  # Custom: M K N dimensions
```

### Memory Management
```bash
cd 04_memory_management/build
./memory_demo             # Runs all memory demonstrations
```

## Troubleshooting

### CUDA Toolkit Not Found
```
CMake Error: Could not find CUDA Toolkit
```
**Solution:** Ensure CUDA Toolkit is installed and `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Compute Capability Mismatch
```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```
**Solution:** Edit CMakeLists.txt and adjust `CUDA_ARCHITECTURES` to match your GPU.

### Runtime Errors
```
CUDA error: no CUDA-capable device is detected
```
**Solution:** Ensure NVIDIA drivers are installed and GPU is recognized:
```bash
nvidia-smi
```

## Performance Tips

1. **Use larger data sets** for better GPU utilization
2. **Profile your code** with Nsight Compute or nvprof
3. **Experiment with block sizes** (typically multiples of 32)
4. **Measure bandwidth** to identify memory bottlenecks
5. **Compare different implementations** (naive vs. optimized)

## Learning Path

**Recommended order for beginners:**

1. Start with **Hello World** to understand basic kernel launches
2. Progress to **Vector Addition** for memory management
3. Try **Matrix Multiplication** for shared memory and optimization
4. Explore **Memory Management** for advanced memory techniques

## Additional Resources

- **Theory Notes:** See `/notes` directory for detailed explanations
- **Main README:** Check the main repository README for extensive resource links
- **CUDA Samples:** NVIDIA's official samples repository for more examples
- **Documentation:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Contributing

Want to add more examples? Contributions are welcome! Please ensure:
- Each example has its own directory with CMakeLists.txt
- Code is well-commented and follows existing style
- Include a README with description and usage instructions
- Examples build successfully with CUDA Toolkit 11.0+

## Support

For issues or questions:
- Check individual project READMEs for specific help
- Review the theory notes in `/notes`
- Open an issue on GitHub
- Consult [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/)
