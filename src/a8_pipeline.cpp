// A8 (Stage 6): CPU-GPU overlapped STAGE pipeline.
//
//   main (decode) -> Q1[20] -> R resize workers -> Q2[K] -> 1 GPU thread (S streams, async)
//                 -> Q3[K] -> T tail workers -> FrameReorder (in-order encode)
//
// Unlike A7 (each worker runs every stage and blocks on the GPU), each stage here has its
// own threads, so CPU and GPU work on different frames at the same time and no CPU tail
// thread ever waits for the GPU.
//
// Buffer ownership: a frame owns one pinned in/out pair (PinnedPairPool, K pairs) from
// resize until its tail has finished; queue pushes transfer ownership. Q2/Q3 have
// capacity K, so pushes into them can never block (enforced with try_push: a failed
// push is an invariant violation and aborts the run).
//
// Liveness: a resize worker acquires a pair BEFORE popping Q1, so frames leave Q1 in
// decode order already owning a pair (no frame waits for a pair after it was dequeued).
// Overtaking is then bounded by (R-1) + (T-1), far below the reorder capacity (60).
//
// Only compiled in CUDA builds (listed with the other GPU sources in CMakeLists.txt).

#include "pipeline.hpp"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bounded_queue.hpp"
#include "config.hpp"
#include "cpu_pipeline.hpp"
#include "cpu_topology.hpp"
#include "filters.hpp"
#include "frame_reorder.hpp"
#include "frame_task.hpp"
#include "gpu_blur.hpp"
#include "gpu_slot_pool.hpp"
#include "pinned_pair_pool.hpp"
#include "trace_manager.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// What travels through Q2 and Q3: no pixels, only the index of the frame's pinned pair.
struct A8Packet {
    int frame_id = -1;
    double video_timestamp = 0.0;
    int pair = -1;
};

// Per-thread time accounting (written only by its own thread; summed after join).
struct A8Stats {
    double busy_ms = 0.0;      // useful work
    double wait_in_ms = 0.0;   // waiting for input (starved by the stage upstream)
    double wait_out_ms = 0.0;  // blocked by the stage downstream / GPU completion / reorder
    long frames = 0;
};

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

bool pin_current_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (*end != '\0') throw std::invalid_argument(std::string(name) + " must be an integer");
    return static_cast<int>(x);
}

A8Stats sum_stats(const std::vector<A8Stats>& v) {
    A8Stats a;
    for (const auto& s : v) {
        a.busy_ms += s.busy_ms;
        a.wait_in_ms += s.wait_in_ms;
        a.wait_out_ms += s.wait_out_ms;
        a.frames += s.frames;
    }
    return a;
}

[[noreturn]] void fatal(const char* who, int id, const char* what) {
    // A lost frame would stall the reorder buffer forever, so any thread error is fatal.
    std::fprintf(stderr, "A8 %s %d failed: %s\n", who, id, what);
    std::_Exit(2);
}

}  // namespace

void run_a8_stage_pipeline(const std::string& input_path, const std::string& output_path,
                           int num_tail, int num_resize) {
    const int S = config::A8_GPU_STREAMS;
    const int slack = env_int("HPC_A8_SLACK", config::A8_POOL_SLACK);
    if (slack < 0) throw std::invalid_argument("HPC_A8_SLACK must be >= 0");
    const int K = num_resize + S + num_tail + slack;
    const bool pin = env_int("HPC_A8_NO_PIN", 0) == 0;

    std::cout << "Executing A8: CPU-GPU Stage Pipeline (" << num_tail << " tail workers, "
              << num_resize << " resize workers, " << S << " GPU streams, " << K
              << " pinned pairs)...\n";
    TraceManager tracer("../traces/a8_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int W = config::TARGET_WIDTH;
    const int H = config::TARGET_HEIGHT;
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           cap.get(cv::CAP_PROP_FPS), cv::Size(W, H), false);
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }
    FrameReorder reorder(writer, tracer, total_frames);

    // Thread placement (decision D3): main, GPU thread, resize workers, tail workers, in
    // cpu_assignment_order(). Capture and writer are opened above, before any pinning,
    // so FFmpeg's own threads keep the default affinity (as in A5).
    const std::vector<int> order = cpu_assignment_order();
    const int n_threads = 2 + num_resize + num_tail;
    auto cpu_of = [&order](int i) { return order[static_cast<size_t>(i) % order.size()]; };
    if (pin) {
        std::cout << "A8 pinning: main -> CPU " << cpu_of(0) << ", gpu -> " << cpu_of(1) << ", resize ->";
        for (int i = 0; i < num_resize; ++i) std::cout << " " << cpu_of(2 + i);
        std::cout << ", tail ->";
        for (int i = 0; i < num_tail; ++i) std::cout << " " << cpu_of(2 + num_resize + i);
        std::cout << "\n";
        if (static_cast<size_t>(n_threads) > order.size())
            std::cerr << "Warning: A8 uses " << n_threads << " threads but only " << order.size()
                      << " CPUs are available; some threads share a CPU.\n";
    } else {
        std::cout << "A8 pinning: disabled (HPC_A8_NO_PIN)\n";
    }

    // GPU setup (excluded from Total Time, reported separately): S slots (stream + device
    // buffers), K pinned pairs (allocated and touched), one warm-up frame per slot.
    const auto setup_t0 = Clock::now();
    GpuSlotPool slots(S, W, H, 3);
    const size_t bytes = static_cast<size_t>(W) * H * 3;
    PinnedPairPool pairs(K, bytes);
    for (int s = 0; s < S; ++s) slots.slot(s).run_sync(pairs.in(0), pairs.out(0));
    const double setup_s = std::chrono::duration<double>(Clock::now() - setup_t0).count();

    std::vector<std::string> slot_names;
    for (int s = 0; s < S; ++s) slot_names.push_back("GPU_S" + std::to_string(s));

    BoundedQueue<FrameTask> q1(20);
    BoundedQueue<A8Packet> q2(static_cast<size_t>(K));
    BoundedQueue<A8Packet> q3(static_cast<size_t>(K));
    std::atomic<int> resize_alive{num_resize};

    A8Stats main_st, gpu_st;
    std::vector<A8Stats> resize_st(num_resize), tail_st(num_tail);

    struct GpuRecord { int frame_id; int slot; int pair; double submit_to_done_ms; GpuBlurTimings t; };
    std::vector<GpuRecord> gpu_records;   // written only by the GPU thread
    gpu_records.reserve(total_frames > 0 ? static_cast<size_t>(total_frames) : 1024);

    // ---- Resize workers: [pool wait] -> pop Q1 -> resize into pinned `in` -> Q2 ----
    auto resize_func = [&](int id) {
        try {
            if (pin && !pin_current_thread(cpu_of(2 + id)))
                std::fprintf(stderr, "Warning: failed to pin A8 resize worker %d\n", id);
            const std::string name = "A8_RESIZE_" + std::to_string(id);
            A8Stats& st = resize_st[id];
            FrameTask task;
            while (true) {
                const auto t0 = Clock::now();
                const int p = pairs.acquire();          // pair first: keeps Q1 order (liveness)
                const auto t1 = Clock::now();
                if (!q1.pop(task)) { pairs.release(p); break; }
                const auto t2 = Clock::now();

                tracer.recordEvent(task.frame_id, "resize", "start", name);
                cv::Mat staged(H, W, CV_8UC3, pairs.in(p));
                filters::resize_into(task.input_frame, staged, W, H);
                tracer.recordEvent(task.frame_id, "resize", "end", name);
                CV_Assert(staged.data == pairs.in(p));  // resize must not reallocate
                task.input_frame.release();
                const auto t3 = Clock::now();

                if (!q2.try_push(A8Packet{task.frame_id, task.video_timestamp, p}))
                    throw std::runtime_error("Q2 push would block (pair-pool invariant violated)");

                st.wait_out_ms += ms_between(t0, t1);   // pool wait = back-pressure
                st.wait_in_ms  += ms_between(t1, t2);   // Q1 empty = decode too slow
                st.busy_ms     += ms_between(t2, t3);
                st.frames++;
            }
            if (resize_alive.fetch_sub(1) == 1) q2.finish();   // last resize worker closes Q2
        } catch (const std::exception& e) { fatal("resize worker", id, e.what()); }
    };

    // ---- GPU thread: async submit up to S frames, complete oldest first -> Q3 ----
    auto gpu_func = [&]() {
        try {
            if (pin && !pin_current_thread(cpu_of(1)))
                std::fprintf(stderr, "Warning: failed to pin A8 GPU thread\n");
            struct InFlight { A8Packet pkt; int slot; Clock::time_point submitted; };
            std::deque<InFlight> inflight;
            bool input_open = true;

            while (true) {
                // 1) Fill free slots. Block on Q2 only when nothing is on the GPU.
                while (input_open && static_cast<int>(inflight.size()) < S) {
                    A8Packet pkt;
                    if (inflight.empty()) {
                        const auto t0 = Clock::now();
                        const bool got = q2.pop(pkt);
                        gpu_st.wait_in_ms += ms_between(t0, Clock::now());
                        if (!got) { input_open = false; break; }   // Q2 finished and drained
                    } else if (!q2.try_pop(pkt)) {
                        break;                                       // nothing ready: complete oldest
                    }
                    const auto t0 = Clock::now();
                    const int s = slots.acquire();                   // never blocks: < S in flight
                    tracer.recordEvent(pkt.frame_id, "blur", "start", slot_names[s]);
                    slots.slot(s).enqueue(pairs.in(pkt.pair), pairs.out(pkt.pair));
                    gpu_st.busy_ms += ms_between(t0, Clock::now());
                    inflight.push_back({pkt, s, t0});
                }
                // Reached with nothing in flight only after Q2 is finished and drained.
                if (inflight.empty()) break;

                // 2) Complete the oldest frame. Only after D2H has finished may the tail
                //    read pairs.out(pair), so Q3 gets the packet only after wait().
                InFlight f = inflight.front();
                inflight.pop_front();
                const auto t0 = Clock::now();
                const GpuBlurTimings t = slots.slot(f.slot).wait();
                const auto t1 = Clock::now();
                tracer.recordEvent(f.pkt.frame_id, "blur", "end", slot_names[f.slot]);
                slots.release(f.slot);
                gpu_st.wait_out_ms += ms_between(t0, t1);
                gpu_st.frames++;
                gpu_records.push_back({f.pkt.frame_id, f.slot, f.pkt.pair, ms_between(f.submitted, t1), t});
                if (!q3.try_push(std::move(f.pkt)))
                    throw std::runtime_error("Q3 push would block (pair-pool invariant violated)");
            }
            q3.finish();
        } catch (const std::exception& e) { fatal("GPU thread", 0, e.what()); }
    };

    // ---- Tail workers: pop Q3 -> CLAHE/Sobel/sharpen from pinned `out` -> release pair -> reorder ----
    auto tail_func = [&](int id) {
        try {
            if (pin && !pin_current_thread(cpu_of(2 + num_resize + id)))
                std::fprintf(stderr, "Warning: failed to pin A8 tail worker %d\n", id);
            const std::string name = "A8_TAIL_" + std::to_string(id);
            A8Stats& st = tail_st[id];
            A8Packet pkt;
            while (true) {
                const auto t0 = Clock::now();
                if (!q3.pop(pkt)) break;
                const auto t1 = Clock::now();

                cv::Mat blurred(H, W, CV_8UC3, pairs.out(pkt.pair));
                FrameTask task;
                task.frame_id = pkt.frame_id;
                task.video_timestamp = pkt.video_timestamp;
                task.output_frame = process_tail_cpu(blurred, pkt.frame_id, tracer, name);
                CV_Assert(task.output_frame.data != blurred.data);  // tail must not alias the pair
                pairs.release(pkt.pair);   // release BEFORE reorder: a waiting frame holds no pair
                const auto t2 = Clock::now();

                reorder.push_and_write(std::move(task));   // in full mode may include encoding
                const auto t3 = Clock::now();

                st.wait_in_ms  += ms_between(t0, t1);
                st.busy_ms     += ms_between(t1, t2);
                st.wait_out_ms += ms_between(t2, t3);
                st.frames++;
            }
        } catch (const std::exception& e) { fatal("tail worker", id, e.what()); }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(gpu_func);
    for (int i = 0; i < num_resize; ++i) threads.emplace_back(resize_func, i);
    for (int i = 0; i < num_tail; ++i) threads.emplace_back(tail_func, i);

    // Pin main after launching the workers (they pin themselves) and before the timer.
    if (pin && !pin_current_thread(cpu_of(0)))
        std::fprintf(stderr, "Warning: failed to pin A8 main thread\n");

    int current_frame_id = 0;
    const auto start_time = Clock::now();

    // ---- Main thread = decode stage ----
    while (true) {
        FrameTask task;   // fresh Mat per read: no clone needed, the frame is never reused
        const auto t0 = Clock::now();
        tracer.recordEvent(current_frame_id, "decode", "start", "A8_MAIN");
        if (!cap.read(task.input_frame) || task.input_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "A8_MAIN");
        const auto t1 = Clock::now();

        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        tracer.recordEvent(current_frame_id, "enqueue", "start", "A8_MAIN");
        q1.push(std::move(task));
        tracer.recordEvent(current_frame_id, "enqueue", "end", "A8_MAIN");
        const auto t2 = Clock::now();

        main_st.busy_ms     += ms_between(t0, t1);
        main_st.wait_out_ms += ms_between(t1, t2);
        main_st.frames++;
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    q1.finish();
    for (auto& t : threads) t.join();

    const bool ok = reorder.finalize(current_frame_id);

    cap.release();
    writer.release();
    const auto end_time = Clock::now();
    const std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    // Per-frame GPU timings (written after timing stops).
    std::vector<double> h2d, kern, d2h, lat;
    std::ofstream csv("../traces/a8_gpu_timings.csv");
    csv << "frame_id,slot,pair,h2d_ms,kernel_ms,d2h_ms,submit_to_done_ms\n";
    for (const auto& r : gpu_records) {
        csv << r.frame_id << ',' << r.slot << ',' << r.pair << ',' << r.t.h2d_ms << ','
            << r.t.kernel_ms << ',' << r.t.d2h_ms << ',' << r.submit_to_done_ms << '\n';
        h2d.push_back(r.t.h2d_ms); kern.push_back(r.t.kernel_ms);
        d2h.push_back(r.t.d2h_ms); lat.push_back(r.submit_to_done_ms);
    }

    std::cout << (ok ? "Pipeline A8 finished successfully.\n" : "Pipeline A8 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
    std::cout << "GPU Setup Time: " << setup_s << " seconds (excluded from Total Time)\n";
    std::printf("GPU blur per frame (median, CUDA events): H2D %.3f ms, kernels %.3f ms, "
                "D2H %.3f ms; submit-to-done (host) %.3f ms\n",
                median_of(h2d), median_of(kern), median_of(d2h), median_of(lat));

    // Stage summary: per-frame means; util = busy / (threads x Total Time).
    const double total_ms = elapsed.count() * 1000.0;
    auto line = [total_ms](const char* role, int n, const A8Stats& s,
                           const char* in_name, const char* out_name) {
        const double f = s.frames > 0 ? static_cast<double>(s.frames) : 1.0;
        std::printf("  %-6s x%-2d busy %6.3f  util %5.1f%%  %-14s %6.3f  %-17s %6.3f\n",
                    role, n, s.busy_ms / f, 100.0 * s.busy_ms / (n * total_ms),
                    in_name, s.wait_in_ms / f, out_name, s.wait_out_ms / f);
    };
    std::printf("A8 stage summary (ms/frame; util = busy / (threads x Total Time)):\n");
    line("decode", 1, main_st, "-", "Q1 push wait");
    line("resize", num_resize, sum_stats(resize_st), "Q1 pop wait", "pool wait");
    line("gpu", 1, gpu_st, "Q2 pop wait", "GPU done wait");
    line("tail", num_tail, sum_stats(tail_st), "Q3 pop wait", "reorder(+encode)");
}