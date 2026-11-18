#pragma once

// CUDA device query (CUDA_INFO command). The old synchronous 3x3 blur path
// (allocate_device_memory / free_device_memory / launch_cuda_blur) was removed in
// Stage 6; GPU blur lives in gpu_blur.hpp (GpuBlur5, exact, error-checked).
void print_cuda_device_info();