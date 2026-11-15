#pragma once
#include <string>

// Stage 5a: GPU 5x5 blur vs filters::apply_blur on synthetic images and on every
// resized video frame. Returns 0 = PASS, 1 = FAIL, 2 = setup/CUDA error.
int run_gpu_blur_check(const std::string& input_path, int tolerance);