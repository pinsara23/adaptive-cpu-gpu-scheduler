#include <iostream>
#include <string>
#include <stdexcept>
#include "pipeline.hpp"
#include <opencv2/core.hpp>
#include <malloc.h>
#include <cstdlib>

#include "gpu_pipeline.hpp"
#include "cpu_topology.hpp"
#include "config.hpp"
#ifdef HPC_HAS_CUDA
#include "gpu_blur.hpp"
#include "gpu_selftest.hpp"
#endif

using namespace std;

// Process-wide allocator configuration, applied identically to every algorithm.
// Stops glibc from returning large frame buffers to the OS on every free, which
// otherwise causes repeated page faults (kernel time) in the per-frame loop.
// Set HPC_NO_MALLOC_TUNE=1 to measure the default allocator instead.
static void configure_allocator() {
    if (std::getenv("HPC_NO_MALLOC_TUNE")) return;
    mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
    mallopt(M_TOP_PAD,        64 * 1024 * 1024);
}

int main(int argc, char** argv) {

    configure_allocator();

    int out = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-encode") config::no_encode = true;
        else argv[out++] = argv[i];
    }
    argc = out;

    // Enforce strict manual multithreading for accurate benchmarking
    cv::setNumThreads(1);

    // Default to A0 if the user doesn't type an algorithm name
    string algo = "A0";
    if (argc > 1) {
        algo = argv[1];
    }

    string input_path = "../videos/input.mp4";
    string output_path = "../output/output_" + algo + ".mp4"; // Saves as output_A0.mp4, output_A1.mp4, etc.

    if (config::no_encode) {
        output_path = "../output/noencode_" + algo + ".mp4";   // never clobber real outputs
        cout << "MODE: compute-only (encode skipped, output not written)\n";
    }

    if (algo == "CUDA_INFO") {
#ifdef HPC_HAS_CUDA
        std::cout << "Querying GPU Hardware...\n";
        print_cuda_device_info();
#else
        std::cout << "Built without CUDA: no GPU information available.\n";
#endif
        return 0;
    }
    else if (algo == "A0") {
        cout << "Executing Algorithm A0: Sequential Baseline...\n";
        run_sequential_pipeline(input_path, output_path);
    }
    else if (algo == "A1") {
        cout << "Executing Algorithm A1: Single Core Pipeline Baseline...\n";
        run_a1_single_core_pipeline(input_path, output_path);
    }
    else if (algo == "A2") {
        int num_workers = (argc > 2) ? stoi(argv[2]) : 2;
        run_a2_smt_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A3") {
        int num_workers = (argc > 2) ? stoi(argv[2]) : 4;
        run_a3_static_multicore_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A4") {
        int num_workers = (argc > 2) ? stoi(argv[2]) : 4;
        run_a4_dynamic_multicore_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A5") {
        int num_workers = (argc > 2) ? stoi(argv[2]) : 4;
        run_a5_pinned_multicore_pipeline(input_path, output_path, num_workers);
    }

#ifdef HPC_HAS_CUDA
    else if (algo == "A6") {
        try { run_a6_gpu_pipeline(input_path, output_path); }
        catch (const std::exception& e) { cerr << "A6 failed: " << e.what() << "\n"; return 2; }
    }
    else if (algo == "A7") {
        int workers = (argc > 2) ? stoi(argv[2]) : 4;
        int slots = (argc > 3) ? stoi(argv[3]) : 2;
        if (workers < 1 || slots < 1){
            cerr << "Error: A7 needs workers >= 1 and slots >= 1 (usage: A7 <workers> <slots>)\n";           
        }
        try { 
            run_a7_multistream_pipeline(input_path, output_path, workers, slots); 
        }catch (const std::exception& e) { 
            cerr << "A7 failed: " << e.what() << "\n"; return 2; 
        }
    }
    else if (algo == "GPU_STREAM_BENCH") {
        int frames      = (argc > 2) ? stoi(argv[2]) : 200;
        int max_streams = (argc > 3) ? stoi(argv[3]) : 8;
        try { run_gpu_stream_bench(frames, max_streams); }
        catch (const std::exception& e) { cerr << e.what() << "\n"; return 2; }
        return 0;
    }
    else if (algo == "A8") {
        // NOTE: reads argv[3], unlike A2-A5 (argv[2]); fixed in Stage 6
        int num_workers = (argc > 3) ? std::stoi(argv[3]) : 8;
        run_a8_hybrid_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "PCIE_BENCH") {
        try { run_pcie_bench(); }
        catch (const std::exception& e) { cerr << e.what() << "\n"; return 2; }
        return 0;
    }
    else if (algo == "BLUR_CHECK") {
        int tolerance = (argc > 2) ? stoi(argv[2]) : 0;
        return run_gpu_blur_check(input_path, tolerance);
    }
#else
    else if (algo == "A6" || algo == "A7" || algo == "A8" ||
             algo == "PCIE_BENCH" || algo == "BLUR_CHECK" || algo == "GPU_STREAM_BENCH") {
        cerr << "Error: " << algo << " needs CUDA, but this build has no CUDA support.\n";
        return -1;
    }
#endif

    else if (algo == "TOPO") {
        print_cpu_topology();
        return 0;
    }
    else {
        cerr << "Error: Unknown algorithm identifier '" << algo << "'\n";
        cerr << "Usage: ./video_pipeline [A0..A8|CUDA_INFO|TOPO|PCIE_BENCH|BLUR_CHECK [tol]]\n";
        return -1;
    }

    return 0;
}