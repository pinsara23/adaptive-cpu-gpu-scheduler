#pragma once
#include <cstddef>
#include <cstdint>

// Opaque CUDA handle types (identical to cudaStream_t / cudaEvent_t). Declared here
// so .cpp files compiled by g++ can use this header without including cuda_runtime.h.
struct CUstream_st;
struct CUevent_st;

struct GpuBlurTimings {
    float  h2d_ms;    // CUDA events: host -> device copy
    float  kernel_ms; // CUDA events: both blur passes
    float  d2h_ms;    // CUDA events: device -> host copy
    double wall_ms;   // host steady_clock around the whole call (launch + wait)
};

// Intended to equal cv::GaussianBlur(src, dst, Size(5,5), 0) for CV_8UC1 / CV_8UC3:
// weights [1 4 6 4 1]/16 (separable), BORDER_REFLECT_101, round-to-nearest.
// Owns pinned host staging buffers, device buffers, one CUDA stream and timing events.
class GpuBlur5 {
public:
    GpuBlur5(int width, int height, int channels);
    ~GpuBlur5();
    GpuBlur5(const GpuBlur5&) = delete;
    GpuBlur5& operator=(const GpuBlur5&) = delete;

    // Pinned (page-locked) host buffers, frame_bytes() each, rows contiguous.
    uint8_t* host_in()  { return h_in_; }
    uint8_t* host_out() { return h_out_; }
    size_t frame_bytes() const { return bytes_; }

    // host_in -> GPU -> host_out on this object's stream, then wait for completion.
    // Synchronous by design (single-stream baseline). Throws std::runtime_error on error.
    GpuBlurTimings run_sync();

private:
    int width_, height_, channels_;
    size_t bytes_;
    uint8_t*  h_in_  = nullptr;
    uint8_t*  h_out_ = nullptr;
    uint8_t*  d_in_  = nullptr;
    uint16_t* d_tmp_ = nullptr;   // horizontal-pass result, exact (max 16*255)
    uint8_t*  d_out_ = nullptr;
    CUstream_st* stream_ = nullptr;
    CUevent_st*  ev_[4]  = {nullptr, nullptr, nullptr, nullptr};
    void release() noexcept;
};

// Stage 5a microbenchmark: pageable vs pinned host memory, H2D and D2H, several sizes.
void run_pcie_bench();