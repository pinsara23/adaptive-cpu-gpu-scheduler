#pragma once
#include <string>

// A0: Sequential Reference
void run_sequential_pipeline(const std::string& input_path, const std::string& output_path);

// A1: Single core Single thread CPU pipeline
void run_a1_single_core_pipeline(const std::string& input_path, const std::string& output_path);

// A3: Static Multicore Scheduling
void run_a3_static_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers);

// A2: Single-Core SMT (2 threads on 1 physical core)
void run_a2_smt_pipeline(const std::string& input_path, const std::string& output_path);