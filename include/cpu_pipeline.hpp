#pragma once
#include <string>
#include <opencv2/core.hpp>
#include "trace_manager.hpp"

// Runs Resize -> Blur -> CLAHE -> Sobel -> Sharpen on one frame (CPU).
// Emits the same start/end trace events as the original inline code.
// `resource` labels the executing worker in the trace (e.g. "CPU_W2").
cv::Mat process_frame_cpu(const cv::Mat& input,
                          int frame_id,
                          TraceManager& tracer,
                          const std::string& resource,
                          int width,
                          int height);