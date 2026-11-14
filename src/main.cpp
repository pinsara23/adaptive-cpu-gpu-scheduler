#include <iostream>
#include <string>
#include "pipeline.hpp"
#include <opencv2/core.hpp>
#include <malloc.h>
#include <cstdlib>

#include "gpu_pipeline.hpp"

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
    
    //Enforce strict manual multithreading for accurate benchmarking
    cv::setNumThreads(1);
    
    // Default to A0 if the user doesn't type an algorithm name
    string algo = "A0"; 
    if (argc > 1) {
        algo = argv[1];
    }

    string input_path = "../videos/input.mp4";
    string output_path = "../output/output_" + algo + ".mp4"; // Saves as output_A0.mp4, output_A1.mp4, etc.

    if (algo == "CUDA_INFO") {
        std::cout << "Querying GPU Hardware...\n";
        print_cuda_device_info();
        return 0; // Exit after printing info
    }
    else if (algo == "A0") {
        cout << "Executing Algorithm A0: Sequential Baseline...\n";
        run_sequential_pipeline(input_path, output_path);
    } 
    else if (algo == "A1"){
        cout << "Executing Algorithm A1: Single Core Pipeline Baseline...\n";
        run_a1_single_core_pipeline(input_path, output_path);
    }
    else if (algo == "A2") {
        run_a2_smt_pipeline(input_path, output_path);
    }
    else if (algo == "A3") {
        int num_workers = 4; // Default number of workers
        if (argc > 2) {
            num_workers = stoi(argv[2]);
        }
        run_a3_static_multicore_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A4") {
        int num_workers = 4; // Default number of workers
        if (argc > 2) {
            num_workers = stoi(argv[2]);
        }
        run_a4_dynamic_multicore_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A5") {
        int num_workers = 4; // Default number of workers
        if (argc > 2) {
            num_workers = stoi(argv[2]);
        }
        run_a5_pinned_multicore_pipeline(input_path, output_path, num_workers);
    }
    else if (algo == "A6") {
        // Make sure to declare void run_a6_gpu_pipeline(const std::string&, const std::string&); above main
        run_a6_gpu_pipeline(input_path, output_path);
    }
    else if (algo == "A7") {
        run_a7_concurrent_pipeline(input_path, output_path);
    }
    else if (algo == "A8") {
        int num_workers = (argc > 3) ? std::stoi(argv[3]) : 8; // Default to 8 if not specified
        run_a8_hybrid_pipeline(input_path, output_path, num_workers);
    }
    else {
        cerr << "Error: Unknown algorithm identifier '" << algo << "'\n";
        cerr << "Usage: ./video_pipeline [A0|A1|A2...]\n";
        return -1;
    }

    return 0;
}