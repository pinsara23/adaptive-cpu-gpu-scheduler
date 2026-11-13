#pragma once
#include "trace_manager.hpp"

void print_cuda_device_info();

// VRAM Memory Management
void allocate_device_memory(unsigned char** d_in, unsigned char** d_out, int width, int height, int channels);
void free_device_memory(unsigned char* d_in, unsigned char* d_out);

// Heterogeneous execution: PCIe transfer, kernel execution, and PCIe return
void launch_cuda_blur(const unsigned char* h_in, unsigned char* h_out, 
                      unsigned char* d_in, unsigned char* d_out, 
                      int width, int height, int channels, 
                      TraceManager& tracer, int frame_id);