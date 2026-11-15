#pragma once
#include <map>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "frame_task.hpp"
#include "trace_manager.hpp"

class FrameReorder {
public:
    // Add total_frames to constructor
    FrameReorder(cv::VideoWriter& writer, TraceManager& tracer, int total_frames, size_t max_capacity = 60);
    void push_and_write(FrameTask&& task);
    
    // decoded_frames: frames actually decoded by the reader (-1 = fall back to container metadata).
    // Returns true only if frames 0..decoded_frames-1 were all emitted exactly once, in order.
    bool finalize(int decoded_frames = -1);
    
    // Frames emitted in order (written, or counted in --no-encode mode). Read after finalize().
    int frames_emitted() const { return frames_emitted_;}

private:
    cv::VideoWriter& writer_;
    TraceManager& tracer_;
    std::map<int, FrameTask> buffer_;
    int next_expected_frame_id_ = 0;
    int frames_emitted_ = 0;
    int total_frames_;
    size_t max_capacity_;
    std::mutex mtx_;
    std::condition_variable cv_capacity_;
};