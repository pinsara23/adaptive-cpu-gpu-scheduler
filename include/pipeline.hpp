#pragma once
#include <string>

// A0: Sequential Reference
void run_sequential_pipeline(const std::string& input_path, const std::string& output_path);

// A1: Single core Single thread CPU pipeline
void run_a1_single_core_pipeline(const std::string& input_path, const std::string& output_path);

// A3: Static Multicore Scheduling
void run_a3_static_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers);

// A2: Single-Core SMT (1 or 2 workers on one physical core; decoder on another core)
void run_a2_smt_pipeline(const std::string& input_path, const std::string& output_path, int num_workers);

// A4: Dynamic Multicore Scheduling (Shared Queue)
void run_a4_dynamic_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers);

// A5: Thread-Pinned Multicore Scheduling
void run_a5_pinned_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers);

void run_a6_gpu_pipeline(const std::string& input_path, const std::string& output_path);

// A7: Fixed multi-stream GPU (num_workers CPU workers share num_slots GPU stream slots)  num_workers = threads, num_slots = cuda streams
void run_a7_multistream_pipeline(const std::string& input_path, const std::string& output_path,
                                 int num_workers, int num_slots);

// A8: CPU-GPU overlapped stage pipeline (decode -> R resize -> GPU thread -> T tail -> reorder)
void run_a8_stage_pipeline(const std::string& input_path, const std::string& output_path,
                           int num_tail, int num_resize);