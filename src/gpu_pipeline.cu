#include "gpu_pipeline.hpp"
#include <iostream>
#include <cuda_runtime.h>

// --- STEP 1: Hardware Query Function ---
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
        std::cout << "  Shared Memory per Block:  " << device_prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Concurrent Kernels:       " << (device_prop.concurrentKernels ? "Yes" : "No") << "\n";
    }
    std::cout << "\nCUDA verification successful.\n";
}

// --- STEP 2: GPU Pipeline Functions ---

// 3x3 Gaussian Blur Device Kernel
__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x < width && y < height) {
        // Handle image borders by directly copying pixels
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            int idx = (y * width + x) * channels;
            output[idx] = input[idx];
            output[idx+1] = input[idx+1];
            output[idx+2] = input[idx+2];
            return;
        }

        // 3x3 Gaussian Kernel weights
        float kernel[3][3] = {
            {1/16.f, 2/16.f, 1/16.f},
            {2/16.f, 4/16.f, 2/16.f},
            {1/16.f, 2/16.f, 1/16.f}
        };

        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int pixel_idx = ((y + ky) * width + (x + kx)) * channels;
                float k = kernel[ky + 1][kx + 1];
                sum[0] += input[pixel_idx] * k;
                sum[1] += input[pixel_idx + 1] * k;
                sum[2] += input[pixel_idx + 2] * k;
            }
        }

        int out_idx = (y * width + x) * channels;
        output[out_idx] = static_cast<unsigned char>(sum[0]);
        output[out_idx+1] = static_cast<unsigned char>(sum[1]);
        output[out_idx+2] = static_cast<unsigned char>(sum[2]);
    }
}

void allocate_device_memory(unsigned char** d_in, unsigned char** d_out, int width, int height, int channels) {
    size_t bytes = width * height * channels * sizeof(unsigned char);
    cudaMalloc((void**)d_in, bytes);
    cudaMalloc((void**)d_out, bytes);
}

void free_device_memory(unsigned char* d_in, unsigned char* d_out) {
    cudaFree(d_in);
    cudaFree(d_out);
}

void launch_cuda_blur(const unsigned char* h_in, unsigned char* h_out, 
                      unsigned char* d_in, unsigned char* d_out, 
                      int width, int height, int channels, 
                      TraceManager& tracer, int frame_id) {
    
    size_t bytes = width * height * channels * sizeof(unsigned char);

    // 1. Host to Device (H2D) Transfer over PCIe
    tracer.recordEvent(frame_id, "pcie_h2d", "start", "GPU_0_MEM");
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    tracer.recordEvent(frame_id, "pcie_h2d", "end", "GPU_0_MEM");

    // 2. Execute GPU Kernel
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    tracer.recordEvent(frame_id, "kernel_blur", "start", "GPU_0_COMPUTE");
    gaussian_blur_kernel<<<blocks, threads>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    tracer.recordEvent(frame_id, "kernel_blur", "end", "GPU_0_COMPUTE");

    // 3. Device to Host (D2H) Transfer over PCIe
    tracer.recordEvent(frame_id, "pcie_d2h", "start", "GPU_0_MEM");
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    tracer.recordEvent(frame_id, "pcie_d2h", "end", "GPU_0_MEM");
}