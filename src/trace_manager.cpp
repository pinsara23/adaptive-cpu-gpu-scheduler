#include "trace_manager.hpp"
#include <iostream>

TraceManager::TraceManager(const std::string& output_file) : filename(output_file) {
    outfile.open(filename);
    if (outfile.is_open()) {
        outfile << "time_ms,frame_id,stage,event,resource\n";
    } else {
        std::cerr << "Failed to open trace file: " << filename << "\n";
    }
    // Use steady_clock
    pipeline_start_time = std::chrono::steady_clock::now();
}

TraceManager::~TraceManager() {
    flush();
    if (outfile.is_open()) {
        outfile.close();
    }
}

void TraceManager::recordEvent(int frame_id, const std::string& stage, const std::string& event_type, const std::string& resource) {
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - pipeline_start_time).count();
    
    std::lock_guard<std::mutex> lock(mtx);
    events.push_back({elapsed_ms, frame_id, stage, event_type, resource});
}

void TraceManager::flush() {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (outfile.is_open() && !events.empty()) {
        // FIX (Issue 4): Sort events by time_ms to ensure strict chronological order
        std::stable_sort(events.begin(), events.end(), [](const TraceEvent& a, const TraceEvent& b) {
            return a.time_ms < b.time_ms;
        });

        for (const auto& ev : events) {
            outfile << ev.time_ms << "," << ev.frame_id << "," << ev.stage << "," << ev.event_type << "," << ev.resource << "\n";
        }
        
        // FIX (Issue 6): Check for file stream errors during write
        if (outfile.bad()) {
            std::cerr << "Error: TraceManager failed to write to " << filename << "\n";
        }
        
        events.clear();
    }
}