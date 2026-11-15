#include "frame_reorder.hpp"
#include "config.hpp"
#include <iostream>

FrameReorder::FrameReorder(cv::VideoWriter& writer, TraceManager& tracer, int total_frames, size_t max_capacity) 
    : writer_(writer), tracer_(tracer), total_frames_(total_frames), max_capacity_(max_capacity) {}

void FrameReorder::push_and_write(FrameTask&& task) {
    std::unique_lock<std::mutex> lock(mtx_);

    cv_capacity_.wait(lock, [this, &task]() {
        return buffer_.size() < max_capacity_ || task.frame_id == next_expected_frame_id_;    });

    if (buffer_.find(task.frame_id) != buffer_.end()) {
        std::cerr << "Error: Duplicate frame ID detected in Reorder Buffer: " << task.frame_id << "\n";
    } else {
        buffer_[task.frame_id] = std::move(task); 
    }

    while (buffer_.find(next_expected_frame_id_) != buffer_.end()) {
        FrameTask& ready_task = buffer_[next_expected_frame_id_];
        
        tracer_.recordEvent(ready_task.frame_id, "encode", "start", "CPU_MAIN");
        if (!config::no_encode) writer_.write(ready_task.output_frame);
        tracer_.recordEvent(ready_task.frame_id, "encode", "end", "CPU_MAIN");
        
        buffer_.erase(next_expected_frame_id_);
        next_expected_frame_id_++;
        frames_emitted_++;
        cv_capacity_.notify_all(); 
    }
}

bool FrameReorder::finalize(int decoded_frames) {
    std::unique_lock<std::mutex> lock(mtx_);
    const int expected = (decoded_frames >= 0) ? decoded_frames : total_frames_;
    const bool ok = (next_expected_frame_id_ == expected) && buffer_.empty();

    if (!ok) {
        std::cerr << "ERROR: reorder finished with missing or stranded frames.\n"
                  << "  expected " << expected
                  << ", emitted in order " << frames_emitted_
                  << ", stranded in buffer " << buffer_.size() << "\n";
        // Still flush stranded frames so the output is inspectable, but they do NOT
        // count as emitted, so the run is reported as failed.
        for (auto& pair : buffer_) {
            std::cerr << "  recovery: writing stranded frame " << pair.first << "\n";
            tracer_.recordEvent(pair.first, "encode", "start", "CPU_MAIN_RECOVERY");
            if (!config::no_encode) writer_.write(pair.second.output_frame);
            tracer_.recordEvent(pair.first, "encode", "end", "CPU_MAIN_RECOVERY");
        }
        buffer_.clear();
    }
    return ok;

}