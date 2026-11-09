#pragma once
#include <string>
# include <vector>
#include <fstream>
#include <mutex>
#include <chrono>

using namespace std;

struct TraceEvent{
    double time_ms;
    int frame_id;
    string stage;
    string event_type;
    string resource;
};

class TraceManager{

    public:
        TraceManager(const string& output_file);
        ~TraceManager();

        void recordEvent(int frame_id, const string& stage, const string& event_type, const string& resource);
        void flush();

    private:
        string filename;
        ofstream outfile;
        vector<TraceEvent> events;
        mutex mtx;
        chrono::high_resolution_clock::time_point pipeline_start_time;
    };