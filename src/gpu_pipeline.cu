#include "gpu_pipeline.hpp"
#include <iostream>
#include <cuda_runtime.h>

void detectAndPrintCudaDevice() {
    std:: cout << "Detecting CUDA devices...\n";

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to get device count. " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return;
    }

    std::cout << "Detected " << deviceCount << " CUDA device(s).\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << "\n"
                  << "  Compute Capability: " << prop.major << "." << prop.minor << "\n"
                  << "  Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n"
                  << "  Multiprocessors (SMs): " << prop.multiProcessorCount << "\n"
                  << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n"
                  << "  Async Engine Count: " << prop.asyncEngineCount << "\n"
                  << "-----------------------------\n";
    }
}