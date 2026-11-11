#include "frame_reorder.hpp"

using namespace std;

FrameReorder::FrameReorder(cv::VideoWriter& writer, TraceManager& tracer) 
    : writer_(writer), tracer_(tracer) {}

void FrameReorder::push_and_write(const FrameTask& task){
    lock_guard<mutex> lock(mtx_);
    buffer_[task.frame_id] = task;

    // write all consecatve frames that are currently ready in the buffer
    while(buffer_.find(next_expected_frame_id_) != buffer_.end()){
        
        FrameTask& ready_task = buffer_[next_expected_frame_id_];

        tracer_.recordEvent(ready_task.frame_id, "encode", "start", "CPU_MAIN");
        writer_.write(ready_task.output_frame);
        tracer_.recordEvent(ready_task.frame_id, "encode", "end", "CPU_MAIN");

        buffer_.erase(next_expected_frame_id_);
        next_expected_frame_id_++;
    }
}