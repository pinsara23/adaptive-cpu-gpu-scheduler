#include "gpu_pipeline.hpp"
#include <iostream>
#include <cuda_runtime.h>

// Hardware query (CUDA_INFO).
void print_cuda_device_info() {
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error_id) << "\n";
        return;
    }

    if (device_count == 0) {
        std::cout << "There are no available device(s) that support CUDA\n";
        return;
    }

    std::cout << "Detected " << device_count << " CUDA Capable device(s)\n";

    for (int dev = 0; dev < device_count; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        std::cout << "\n=== Device " << dev << ": " << device_prop.name << " ===\n";
        std::cout << "  Compute Capability:       " << device_prop.major << "." << device_prop.minor << "\n";
        std::cout << "  Total VRAM:               " << device_prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Multiprocessors (SMs):    " << device_prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block:    " << device_prop.maxThreadsPerBlock << "\n";
        std::cout << "  Shared Memory per Block:  " << device_prop.sharedMemPerBlock / 1024 <<" KB\n";
        std::cout << "  Concurrent Kernels:       " << (device_prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  Copy engines (async):     " << device_prop.asyncEngineCount << "\n";
        std::cout << "  L2 cache:                 " << device_prop.l2CacheSize / 1024 << " KB\n";
        std::cout << "  Memory bus width:         " << device_prop.memoryBusWidth << " bit\n";
        std::cout << "  Memory clock (reported):  " << device_prop.memoryClockRate / 1000 << "MHz\n";
        std::cout << "  GPU clock (reported max): " << device_prop.clockRate / 1000 << " MHz\n";
        std::cout << "  Registers per SM:         " << device_prop.regsPerMultiprocessor << "\n";
        std::cout << "  Shared memory per SM:     " << device_prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
        std::cout << "  Max threads per SM:       " << device_prop.maxThreadsPerMultiProcessor<< "\n";
        std::cout << "  Warp size:                " << device_prop.warpSize << "\n";
    }
    std::cout << "\nCUDA verification successful.\n";
}