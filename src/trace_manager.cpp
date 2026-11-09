#include "trace_manager.hpp"
#include <iostream>

TraceManager::TraceManager(const std::string& output_file) : filename(output_file) {
    outfile.open(filename);
    if (outfile.is_open()) {
        outfile << "time_ms,frame_id,stage,event,resource\n";
    } else {
        std::cerr << "Failed to open trace file: " << filename << "\n";
    }
    pipeline_start_time = std::chrono::high_resolution_clock::now();
}

TraceManager::~TraceManager() {
    flush();
    if (outfile.is_open()) {
        outfile.close();
    }
}

void TraceManager::recordEvent(int frame_id, const std::string& stage, const std::string& event_type, const std::string& resource) {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - pipeline_start_time).count();
    
    std::lock_guard<std::mutex> lock(mtx);
    events.push_back({elapsed_ms, frame_id, stage, event_type, resource});
}

void TraceManager::flush() {
    std::lock_guard<std::mutex> lock(mtx);
    if (outfile.is_open()) {
        for (const auto& ev : events) {
            outfile << ev.time_ms << "," << ev.frame_id << "," << ev.stage << "," << ev.event_type << "," << ev.resource << "\n";
        }
        events.clear();
    }
}