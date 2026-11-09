#pragma once
#include <opencv2/opencv.hpp>

namespace filters {
    cv::Mat resize_frame(const cv::Mat& input, int width, int height);
    cv::Mat apply_blur(const cv::Mat& input);
    cv::Mat apply_clahe(const cv::Mat& input);
    cv::Mat apply_sobel(const cv::Mat& input);
    cv::Mat apply_sharpen(const cv::Mat& input);


}
