#include "frame_reorder.hpp"
#include <iostream>

FrameReorder::FrameReorder(cv::VideoWriter& writer, TraceManager& tracer, int total_frames, size_t max_capacity) 
    : writer_(writer), tracer_(tracer), total_frames_(total_frames), max_capacity_(max_capacity) {}

void FrameReorder::push_and_write(FrameTask&& task) {
    std::unique_lock<std::mutex> lock(mtx_);

    cv_capacity_.wait(lock, [this]() {
        return buffer_.size() < max_capacity_;
    });

    if (buffer_.find(task.frame_id) != buffer_.end()) {
        std::cerr << "Error: Duplicate frame ID detected in Reorder Buffer: " << task.frame_id << "\n";
    } else {
        buffer_[task.frame_id] = std::move(task); 
    }

    while (buffer_.find(next_expected_frame_id_) != buffer_.end()) {
        FrameTask& ready_task = buffer_[next_expected_frame_id_];
        
        tracer_.recordEvent(ready_task.frame_id, "encode", "start", "CPU_MAIN");
        writer_.write(ready_task.output_frame);
        tracer_.recordEvent(ready_task.frame_id, "encode", "end", "CPU_MAIN");
        
        buffer_.erase(next_expected_frame_id_);
        next_expected_frame_id_++;
        
        cv_capacity_.notify_all(); 
    }
}

void FrameReorder::finalize() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (next_expected_frame_id_ < total_frames_) {
        std::cerr << "Warning: Pipeline finished but missing frames detected.\n"
                  << "Expected total: " << total_frames_ << ", Last successfully ordered: " << next_expected_frame_id_ << "\n";
        
        // Flush any stranded frames that were waiting for the missing frame
        for (auto& pair : buffer_) {
            std::cerr << "Recovery: Forced writing stranded out-of-order frame " << pair.first << "\n";
            tracer_.recordEvent(pair.first, "encode", "start", "CPU_MAIN_RECOVERY");
            writer_.write(pair.second.output_frame);
            tracer_.recordEvent(pair.first, "encode", "end", "CPU_MAIN_RECOVERY");
        }
        buffer_.clear();
    }
}