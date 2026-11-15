#pragma once
// CUDA error checking. Include ONLY from .cu files (needs cuda_runtime.h).
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

// Every CUDA runtime call returns cudaError_t. Ignoring it hides failures: a failed
// cudaMalloc leaves a null pointer and later work faults or silently does nothing.
// CUDA_CHECK turns any failure into an exception naming the file, line and call.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                cudaGetErrorString(err_) + " at " + __FILE__ + ":" +           \
                std::to_string(__LINE__) + " in " #call);                      \
        }                                                                      \
    } while (0)

// For destructors and cleanup, which must not throw: report and continue.
#define CUDA_WARN(call)                                                        \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA warning: %s at %s:%d in %s\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call); \
        }                                                                      \
    } while (0)
