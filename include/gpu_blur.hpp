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
    double wall_ms;   // host clock around run_sync (0 when using enqueue/wait)
};

// Page-locked (pinned) host memory, owned. Needed for truly asynchronous copies.
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t bytes);
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    uint8_t* data() { return ptr_; }
    size_t size() const { return bytes_; }
private:
    uint8_t* ptr_ = nullptr;
    size_t bytes_;
};

// Intended to equal cv::GaussianBlur(src, dst, Size(5,5), 0) for CV_8UC1 / CV_8UC3:
// weights [1 4 6 4 1]/16 (separable), BORDER_REFLECT_101, round-to-nearest.
// Owns device buffers, one CUDA stream and timing events; optionally its own pinned
// host buffers (own_host_buffers=false: callers pass their own pinned host pointers).
class GpuBlur5 {
public:
    GpuBlur5(int width, int height, int channels, bool own_host_buffers = true);
    ~GpuBlur5();
    GpuBlur5(const GpuBlur5&) = delete;
    GpuBlur5& operator=(const GpuBlur5&) = delete;

    // Own pinned host buffers (nullptr if own_host_buffers=false), frame_bytes() each.
    uint8_t* host_in()  { return h_in_; }
    uint8_t* host_out() { return h_out_; }
    size_t frame_bytes() const { return bytes_; }

    // Synchronous: copy in, blur, copy out, wait. Throws std::runtime_error on error.
    GpuBlurTimings run_sync();                                  // own host buffers
    GpuBlurTimings run_sync(const uint8_t* h_in, uint8_t* h_out); // caller's pinned buffers

    // Asynchronous building blocks. enqueue() returns immediately; wait() blocks until
    // this stream's work is done and returns CUDA-event timings (wall_ms = 0).
    // h_in/h_out must stay valid and untouched until wait() returns. Call wait() only
    // after an enqueue().
    void enqueue(const uint8_t* h_in, uint8_t* h_out);
    GpuBlurTimings wait();

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

// Stage 5c (option C): GPU-only stream scaling. One host thread, frames round-robin
// over S = 1, 2, 4, ... max_streams streams, no CPU stages.
void run_gpu_stream_bench(int frames, int max_streams);