#include "filters.hpp"

namespace filters {
    cv::Mat resize_frame(const cv::Mat& input, int width, int height) {
        cv::Mat output;
        cv::resize(input, output, cv::Size(width, height));
        return output;
    }

    cv::Mat apply_blur(const cv::Mat& input) {
        cv::Mat output;
        cv::GaussianBlur(input, output, cv::Size(5, 5), 0);
        return output;
    }

    cv::Mat apply_clahe(const cv::Mat& input) {
        cv::Mat gray, output;
        if (input.channels() == 3) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, output);
        return output;
    }

    cv::Mat apply_sobel(const cv::Mat& input) {
        cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, output;
        // Compute gradients along X and Y
        cv::Sobel(input, grad_x, CV_16S, 1, 0, 3);
        cv::Sobel(input, grad_y, CV_16S, 0, 1, 3);
        // Convert back to 8-bit
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        // Blend the two gradients
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output);
        return output;
    }

    cv::Mat apply_sharpen(const cv::Mat& input) {
        cv::Mat output;
        // Simple Laplacian-based sharpening kernel
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                           0, -1,  0,
                          -1,  5, -1,
                           0, -1,  0);
        cv::filter2D(input, output, input.depth(), kernel);
        return output;
    }
}