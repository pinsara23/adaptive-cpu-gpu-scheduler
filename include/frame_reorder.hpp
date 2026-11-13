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
    
    // Ensure no frames are stranded during shutdown
    void finalize(); 

private:
    cv::VideoWriter& writer_;
    TraceManager& tracer_;
    std::map<int, FrameTask> buffer_;
    int next_expected_frame_id_ = 0;
    int total_frames_;
    size_t max_capacity_;
    std::mutex mtx_;
    std::condition_variable cv_capacity_;
};