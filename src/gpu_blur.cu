// Stage 5a/5c: exact 5x5 Gaussian blur on the GPU, PCIe benchmark, stream benchmark.
#include "gpu_blur.hpp"
#include "cuda_check.hpp"
#include "config.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// ------------------------------------------------------------------ kernels

// BORDER_REFLECT_101 (OpenCV default): mirror without repeating the edge pixel.
//   index:  -2 -1 | 0 1 2 ... n-1 | n   n+1
//   maps:    2  1 | 0 1 2 ... n-1 | n-2 n-3
// One reflection is enough because the radius (2) is smaller than n (n >= 3, host-checked).
__device__ __forceinline__ int reflect101(int i, int n) {
    if (i < 0)  i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i;
}

// Pass 1: horizontal [1 4 6 4 1] on interleaved 8-bit pixels -> exact uint16 sums.
__global__ void blur5_h(const uint8_t* __restrict__ src, uint16_t* __restrict__ tmp,
                        int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;       // partial blocks at right/bottom edges

    const int stride = width * channels;          // contiguous rows
    const uint8_t* row = src + static_cast<size_t>(y) * stride;
    const int c0 = reflect101(x - 2, width) * channels;
    const int c1 = reflect101(x - 1, width) * channels;
    const int c2 = x * channels;
    const int c3 = reflect101(x + 1, width) * channels;
    const int c4 = reflect101(x + 2, width) * channels;
    uint16_t* out = tmp + static_cast<size_t>(y) * stride + c2;

    for (int c = 0; c < channels; ++c) {
        out[c] = static_cast<uint16_t>(row[c0 + c] + 4 * row[c1 + c] + 6 * row[c2 + c]
                                     + 4 * row[c3 + c] + row[c4 + c]);   // max 16*255
    }
}

// Pass 2: vertical [1 4 6 4 1] on the uint16 sums -> S (max 256*255), round S/256.
__global__ void blur5_v(const uint16_t* __restrict__ tmp, uint8_t* __restrict__ dst,
                        int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t stride = static_cast<size_t>(width) * channels;
    const size_t col = static_cast<size_t>(x) * channels;
    const size_t r0 = reflect101(y - 2, height) * stride;
    const size_t r1 = reflect101(y - 1, height) * stride;
    const size_t r2 = static_cast<size_t>(y) * stride;
    const size_t r3 = reflect101(y + 1, height) * stride;
    const size_t r4 = reflect101(y + 2, height) * stride;

    for (int c = 0; c < channels; ++c) {
        const size_t i = col + c;
        const unsigned s = tmp[r0 + i] + 4u * tmp[r1 + i] + 6u * tmp[r2 + i]
                         + 4u * tmp[r3 + i] + tmp[r4 + i];
        dst[r2 + i] = static_cast<uint8_t>((s + 128u) >> 8);   // round half up; max 255
    }
}

// ------------------------------------------------------------------ PinnedBuffer

PinnedBuffer::PinnedBuffer(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaMallocHost(&ptr_, bytes_));
}

PinnedBuffer::~PinnedBuffer() {
    if (ptr_) CUDA_WARN(cudaFreeHost(ptr_));
}

// ------------------------------------------------------------------ GpuBlur5

GpuBlur5::GpuBlur5(int width, int height, int channels, bool own_host_buffers)
    : width_(width), height_(height), channels_(channels),
      bytes_(static_cast<size_t>(width) * height * channels) {
    if (width < 3 || height < 3 || (channels != 1 && channels != 3))
        throw std::invalid_argument("GpuBlur5: need width,height >= 3 and channels 1 or 3");
    try {
        if (own_host_buffers) {
            CUDA_CHECK(cudaMallocHost(&h_in_,  bytes_));   // pinned
            CUDA_CHECK(cudaMallocHost(&h_out_, bytes_));   // pinned
        }
        CUDA_CHECK(cudaMalloc(&d_in_,  bytes_));
        CUDA_CHECK(cudaMalloc(&d_tmp_, bytes_ * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_out_, bytes_));
        // Non-blocking: this stream does not implicitly wait for the legacy default
        // stream (which the old A8 code still uses).
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));
    } catch (...) {
        release();   // don't leak what was allocated before the failure
        throw;
    }
}

GpuBlur5::~GpuBlur5() { release(); }

void GpuBlur5::release() noexcept {
    for (auto& e : ev_) { if (e) CUDA_WARN(cudaEventDestroy(e)); e = nullptr; }
    if (stream_) CUDA_WARN(cudaStreamDestroy(stream_));
    if (d_out_)  CUDA_WARN(cudaFree(d_out_));
    if (d_tmp_)  CUDA_WARN(cudaFree(d_tmp_));
    if (d_in_)   CUDA_WARN(cudaFree(d_in_));
    if (h_out_)  CUDA_WARN(cudaFreeHost(h_out_));
    if (h_in_)   CUDA_WARN(cudaFreeHost(h_in_));
    stream_ = nullptr; d_out_ = nullptr; d_tmp_ = nullptr; d_in_ = nullptr;
    h_out_ = nullptr; h_in_ = nullptr;
}

void GpuBlur5::enqueue(const uint8_t* h_in, uint8_t* h_out) {
    if (!h_in || !h_out) throw std::invalid_argument("GpuBlur5::enqueue: null host buffer");
    const dim3 block(32, 8);   // 32 threads along x = one warp per row segment (coalesced)
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_in_, h_in, bytes_, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
    blur5_h<<<grid, block, 0, stream_>>>(d_in_, d_tmp_, width_, height_, channels_);
    CUDA_CHECK(cudaGetLastError());   // launch errors (bad configuration)
    blur5_v<<<grid, block, 0, stream_>>>(d_tmp_, d_out_, width_, height_, channels_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out_, bytes_, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(ev_[3], stream_));
}

GpuBlurTimings GpuBlur5::wait() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // execution errors surface here
    GpuBlurTimings t{};
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms,    ev_[0], ev_[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, ev_[1], ev_[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms,    ev_[2], ev_[3]));
    return t;
}

GpuBlurTimings GpuBlur5::run_sync(const uint8_t* h_in, uint8_t* h_out) {
    const auto t0 = std::chrono::steady_clock::now();
    enqueue(h_in, h_out);
    GpuBlurTimings t = wait();
    t.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return t;
}

GpuBlurTimings GpuBlur5::run_sync() { return run_sync(h_in_, h_out_); }

// ------------------------------------------------------------------ helpers

namespace {
struct Stat { double median, min, max; };
Stat summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}
}  // namespace

// ------------------------------------------------------------------ PCIe bench

void run_pcie_bench() {
    constexpr size_t kMax = 64ull << 20;
    struct Case { size_t bytes; const char* label; };
    const Case cases[] = {
        {1280ull * 720,     "720p gray 0.92MB"},
        {1280ull * 720 * 3, "720p BGR 2.76MB"},
        {kMax,              "64 MiB"},
    };
    constexpr int kWarmup = 10, kReps = 100;

    std::vector<uint8_t> pageable(kMax, 1);   // written once so every page really exists
    uint8_t* pinned = nullptr;
    uint8_t* dev = nullptr;
    cudaStream_t s = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned, kMax));
    std::memset(pinned, 1, kMax);
    CUDA_CHECK(cudaMalloc(&dev, kMax));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    std::printf("PCIe transfer benchmark: %d warm-up + %d timed copies per row\n", kWarmup, kReps);
    std::printf("wall  = host clock around copy + sync (what a pipeline thread pays)\n");
    std::printf("event = GPU-side; for pageable memory it may miss the driver's staging copy\n\n");
    std::printf("%-17s %-8s %-3s | %9s %9s %9s | %9s | %7s\n", "size", "host mem", "dir",
                "wall med", "wall min", "wall max", "event med", "GB/s");

    for (const Case& cs : cases) {
        for (int pin = 0; pin < 2; ++pin) {
            uint8_t* host = pin ? pinned : pageable.data();
            for (int h2d = 1; h2d >= 0; --h2d) {
                std::vector<double> wall, ev;
                for (int i = 0; i < kWarmup + kReps; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    CUDA_CHECK(cudaEventRecord(e0, s));
                    if (h2d) CUDA_CHECK(cudaMemcpyAsync(dev, host, cs.bytes, cudaMemcpyHostToDevice, s));
                    else     CUDA_CHECK(cudaMemcpyAsync(host, dev, cs.bytes, cudaMemcpyDeviceToHost, s));
                    CUDA_CHECK(cudaEventRecord(e1, s));
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    const auto t1 = std::chrono::steady_clock::now();
                    if (i < kWarmup) continue;
                    float ms = 0.f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                    wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    ev.push_back(ms);
                }
                const Stat w = summarize(wall), e = summarize(ev);
                const double gbps = cs.bytes / (w.median * 1e-3) / 1e9;   // decimal GB/s
                std::printf("%-17s %-8s %-3s | %9.3f %9.3f %9.3f | %9.3f | %7.2f\n", cs.label,
                            pin ? "pinned" : "pageable", h2d ? "H2D" : "D2H",
                            w.median, w.min, w.max, e.median, gbps);
                std::fflush(stdout);
            }
        }
    }
    CUDA_WARN(cudaEventDestroy(e0));
    CUDA_WARN(cudaEventDestroy(e1));
    CUDA_WARN(cudaStreamDestroy(s));
    CUDA_WARN(cudaFree(dev));
    CUDA_WARN(cudaFreeHost(pinned));
}

// ------------------------------------------------------------------ stream bench

void run_gpu_stream_bench(int frames, int max_streams) {
    if (frames < 1 || max_streams < 1)
        throw std::invalid_argument("GPU_STREAM_BENCH: need frames >= 1 and max_streams >= 1");
    const int W = config::TARGET_WIDTH, H = config::TARGET_HEIGHT, C = 3;
    const size_t bytes = static_cast<size_t>(W) * H * C;
    constexpr int kReps = 5;

    std::vector<int> counts;
    for (int s = 1; s <= max_streams; s *= 2) counts.push_back(s);
    if (counts.back() != max_streams) counts.push_back(max_streams);

    // One pinned input/output pair per stream, filled once with a fixed pattern.
    std::vector<std::unique_ptr<PinnedBuffer>> hin, hout;
    for (int i = 0; i < max_streams; ++i) {
        hin.push_back(std::make_unique<PinnedBuffer>(bytes));
        hout.push_back(std::make_unique<PinnedBuffer>(bytes));
        uint8_t* p = hin.back()->data();
        for (size_t j = 0; j < bytes; ++j) p[j] = static_cast<uint8_t>((j * 31 + i * 17) & 0xFF);
    }

    std::printf("GPU stream benchmark: %d frames per run, %dx%d BGR, %d timed runs after warm-up\n",
                frames, W, H, kReps);
    std::printf("One host thread; frame i -> stream i %% S; a stream's buffers are reused only\n");
    std::printf("after its previous frame finished. Per-op event times include waiting for a\n");
    std::printf("busy copy engine when streams overlap; compare throughput, not per-op times.\n\n");
    std::printf("%7s | %9s %9s %9s | %8s | %8s | %6s | %6s %6s %6s\n", "streams", "total ms",
                "min", "max", "ms/frame", "frames/s", "vs S=1", "H2D", "kern", "D2H");

    double base = 0.0;
    for (int S : counts) {
        std::vector<std::unique_ptr<GpuBlur5>> slots;
        for (int s = 0; s < S; ++s) slots.push_back(std::make_unique<GpuBlur5>(W, H, C, false));
        for (int s = 0; s < S; ++s) slots[s]->enqueue(hin[s]->data(), hout[s]->data());   // warm-up
        for (int s = 0; s < S; ++s) slots[s]->wait();

        std::vector<double> totals, h2d, kern, d2h;
        for (int rep = 0; rep < kReps; ++rep) {
            std::vector<char> busy(S, 0);
            auto collect = [&](const GpuBlurTimings& t) {
                h2d.push_back(t.h2d_ms); kern.push_back(t.kernel_ms); d2h.push_back(t.d2h_ms);
            };
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < frames; ++i) {
                const int s = i % S;
                if (busy[s]) collect(slots[s]->wait());   // stream s free again
                slots[s]->enqueue(hin[s]->data(), hout[s]->data());
                busy[s] = 1;
            }
            for (int s = 0; s < S; ++s) if (busy[s]) collect(slots[s]->wait());
            totals.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
        const Stat t = summarize(totals);
        if (S == 1) base = t.median;
        const double per_frame = t.median / frames;
        std::printf("%7d | %9.2f %9.2f %9.2f | %8.3f | %8.0f | %5.2fx | %6.3f %6.3f %6.3f\n",
                    S, t.median, t.min, t.max, per_frame, 1000.0 / per_frame,
                    base > 0 ? base / t.median : 1.0,
                    summarize(h2d).median, summarize(kern).median, summarize(d2h).median);
        std::fflush(stdout);
    }
}