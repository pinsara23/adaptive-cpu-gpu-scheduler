#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <chrono>
#include <algorithm> // Required for std::stable_sort

struct TraceEvent {
    double time_ms;
    int frame_id;
    std::string stage;
    std::string event_type; // "start" or "end"
    std::string resource;
};

class TraceManager {
public:
    TraceManager(const std::string& output_file);
    ~TraceManager();

    void recordEvent(int frame_id, const std::string& stage, const std::string& event_type, const std::string& resource);
    void flush();

private:
    std::string filename;
    std::ofstream outfile;
    std::vector<TraceEvent> events;
    std::mutex mtx; 
    
    // FIX (Issue 5): Use steady_clock to guarantee monotonic timing
    std::chrono::steady_clock::time_point pipeline_start_time; 
};