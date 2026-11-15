#include "gpu_selftest.hpp"
#include "gpu_blur.hpp"
#include "filters.hpp"
#include "config.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

struct DiffStats {
    long long elements = 0, differing = 0;
    double sum_abs = 0.0;
    int max_abs = 0;

    void add(const cv::Mat& cpu, const cv::Mat& gpu) {
        cv::Mat diff;
        cv::absdiff(cpu, gpu, diff);
        cv::Mat flat = diff.reshape(1);          // one channel: every byte counted
        double mx = 0.0;
        cv::minMaxLoc(flat, nullptr, &mx);
        max_abs = std::max(max_abs, static_cast<int>(mx));
        differing += cv::countNonZero(flat);
        sum_abs += cv::sum(flat)[0];
        elements += static_cast<long long>(flat.total());
    }
    void merge(const DiffStats& o) {
        elements += o.elements; differing += o.differing;
        sum_abs += o.sum_abs;   max_abs = std::max(max_abs, o.max_abs);
    }
};

// Print the first n differing elements: location, channel and both values.
void print_first_diffs(const cv::Mat& cpu, const cv::Mat& gpu, int n) {
    const int ch = cpu.channels();
    cv::Mat diff;
    cv::absdiff(cpu, gpu, diff);
    std::vector<cv::Point> pts;
    cv::findNonZero(diff.reshape(1), pts);
    for (int i = 0; i < static_cast<int>(pts.size()) && i < n; ++i) {
        const int y = pts[i].y, xb = pts[i].x;
        std::printf("      (x=%d, y=%d, c=%d) cpu=%d gpu=%d\n", xb / ch, y, xb % ch,
                    cpu.ptr<uchar>(y)[xb], gpu.ptr<uchar>(y)[xb]);
    }
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Odd sizes exercise border mirroring and partial thread blocks, which 1280x720 hides.
bool synthetic_tests(int tolerance) {
    const cv::Size sizes[] = {{37, 23}, {61, 17}, {1280, 720}};
    const int channel_counts[] = {1, 3};
    cv::RNG rng(12345);                      // fixed seed: reproducible
    bool ok = true;
    for (const cv::Size& sz : sizes) {
        for (int ch : channel_counts) {
            GpuBlur5 g(sz.width, sz.height, ch);
            const int type = CV_MAKETYPE(CV_8U, ch);
            cv::Mat in(sz, type, g.host_in());    // header over pinned memory, no copy
            rng.fill(in, cv::RNG::UNIFORM, 0, 256);
            g.run_sync();
            cv::Mat gpu(sz, type, g.host_out());
            cv::Mat cpu = filters::apply_blur(in);
            DiffStats d;
            d.add(cpu, gpu);
            const bool pass = d.max_abs <= tolerance;
            std::printf("  synthetic %4dx%-4d ch=%d  max_abs=%d  differing=%lld/%lld  %s\n",
                        sz.width, sz.height, ch, d.max_abs, d.differing, d.elements,
                        pass ? "ok" : "FAIL");
            if (!pass) { print_first_diffs(cpu, gpu, 5); ok = false; }
        }
    }
    return ok;
}

}  // namespace

int run_gpu_blur_check(const std::string& input_path, int tolerance) {
    try {
        std::printf("GPU blur check (pass rule: max abs diff <= %d)\n", tolerance);
        std::printf("[1] Synthetic random images\n");
        const bool synth_ok = synthetic_tests(tolerance);

        const int W = config::TARGET_WIDTH, H = config::TARGET_HEIGHT;
        std::printf("[2] Video frames: %s, resized to %dx%d (BGR)\n", input_path.c_str(), W, H);
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "Error: cannot open %s\n", input_path.c_str());
            return 2;
        }

        GpuBlur5 g(W, H, 3);
        cv::Mat staged(H, W, CV_8UC3, g.host_in());    // pinned input, wrapped
        cv::Mat gpu_out(H, W, CV_8UC3, g.host_out());  // pinned output, wrapped

        DiffStats total;
        std::vector<double> cpu_ms, h2d_ms, kern_ms, d2h_ms, wall_ms;
        int frames = 0, failing_frames = 0;
        cv::Mat frame;

        while (cap.read(frame)) {
            if (frame.empty()) break;
            // Same resize function as the real pipeline; copyTo reuses the pinned buffer.
            filters::resize_frame(frame, W, H).copyTo(staged);
            CV_Assert(staged.data == g.host_in());

            const auto c0 = std::chrono::steady_clock::now();
            cv::Mat cpu_out = filters::apply_blur(staged);
            const auto c1 = std::chrono::steady_clock::now();
            const GpuBlurTimings t = g.run_sync();

            DiffStats d;
            d.add(cpu_out, gpu_out);
            if (d.max_abs > tolerance) {
                if (failing_frames == 0) {
                    std::printf("  first failing frame: %d (max_abs=%d, differing=%lld)\n",
                                frames, d.max_abs, d.differing);
                    print_first_diffs(cpu_out, gpu_out, 5);
                }
                ++failing_frames;
            }
            total.merge(d);

            cpu_ms.push_back(std::chrono::duration<double, std::milli>(c1 - c0).count());
            h2d_ms.push_back(t.h2d_ms);
            kern_ms.push_back(t.kernel_ms);
            d2h_ms.push_back(t.d2h_ms);
            wall_ms.push_back(t.wall_ms);
            ++frames;
        }

        const bool video_ok = frames > 0 && failing_frames == 0;
        std::printf("  frames compared:      %d (container says %d)\n", frames,
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)));
        std::printf("  max abs diff:         %d\n", total.max_abs);
        std::printf("  differing elements:   %lld / %lld (%.6f%%)\n", total.differing,
                    total.elements, total.elements ? 100.0 * total.differing / total.elements : 0.0);
        std::printf("  mean abs diff:        %.6f\n",
                    total.elements ? total.sum_abs / total.elements : 0.0);
        std::printf("  failing frames:       %d\n", failing_frames);
        std::printf("[3] Per-frame medians (GPU parts = CUDA events; GPU wall = host clock)\n");
        std::printf("  CPU blur (1 thread):  %.3f ms\n", median(cpu_ms));
        std::printf("  GPU H2D:              %.3f ms\n", median(h2d_ms));
        std::printf("  GPU kernels (h+v):    %.3f ms\n", median(kern_ms));
        std::printf("  GPU D2H:              %.3f ms\n", median(d2h_ms));
        std::printf("  GPU wall total:       %.3f ms\n", median(wall_ms));

        const bool ok = synth_ok && video_ok;
        std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "BLUR_CHECK error: %s\n", e.what());
        return 2;
    }
}