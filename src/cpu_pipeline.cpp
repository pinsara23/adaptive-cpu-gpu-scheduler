#include "cpu_pipeline.hpp"
#include "filters.hpp"

cv::Mat process_frame_cpu(const cv::Mat& input,
                          int frame_id,
                          TraceManager& tracer,
                          const std::string& resource,
                          int width,
                          int height) {
    tracer.recordEvent(frame_id, "resize", "start", resource);
    cv::Mat resized = filters::resize_frame(input, width, height);
    tracer.recordEvent(frame_id, "resize", "end", resource);

    tracer.recordEvent(frame_id, "blur", "start", resource);
    cv::Mat blurred = filters::apply_blur(resized);
    tracer.recordEvent(frame_id, "blur", "end", resource);

    tracer.recordEvent(frame_id, "clahe", "start", resource);
    cv::Mat clahe = filters::apply_clahe(blurred);
    tracer.recordEvent(frame_id, "clahe", "end", resource);

    tracer.recordEvent(frame_id, "sobel", "start", resource);
    cv::Mat sobel = filters::apply_sobel(clahe);
    tracer.recordEvent(frame_id, "sobel", "end", resource);

    tracer.recordEvent(frame_id, "sharpen", "start", resource);
    cv::Mat sharp = filters::apply_sharpen(sobel);
    tracer.recordEvent(frame_id, "sharpen", "end", resource);

    return sharp;
}