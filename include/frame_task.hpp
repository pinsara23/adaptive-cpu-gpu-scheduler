#pragma once
#include <opencv2/opencv.hpp>

struct FrameTask{
    int frame_id;
    double video_timestamp; // location in origianl video
    cv::Mat input_frame; // the actual frame data
    cv::Mat output_frame; // the processed frame data
};