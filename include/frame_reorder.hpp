#pragma once
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "frame_task.hpp"
#include "trace_manager.hpp"

class FrameReorder{
    public: 
        FrameReorder(cv::VideoWriter& writer, TraceManager& tracer);
        void push_and_write(const FrameTask& task);

    private:
        cv::VideoWriter& writer_;
        TraceManager& tracer_;
        std::map<int, FrameTask> buffer_;
        int next_expected_frame_id_ = 0;
        std::mutex mtx_;
};

