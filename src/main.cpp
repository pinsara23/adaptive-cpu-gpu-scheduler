#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "filters.hpp"

int main() {
    std::cout << "Starting Sequential Video Pipeline (Baseline)..." << std::endl;

    // Open input video
    std::string input_path = "../videos/input.mp4";
    cv::VideoCapture cap(input_path);
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video at " << input_path << "\n";
        return -1;
    }

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    // We will standardise the workload to 1280x720 resolution
    int target_width = 1280;
    int target_height = 720;

    // Setup VideoWriter (isColor = false, because our pipeline output is grayscale)
    std::string output_path = "../output/output_sequential.mp4";
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           fps, cv::Size(target_width, target_height), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not create output video writer.\n";
        return -1;
    }

    int frames_processed = 0;
    cv::Mat frame;

    // Start timing using high resolution clock
    auto start_time = std::chrono::high_resolution_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Execute Pipeline
        cv::Mat resized   = filters::resize_frame(frame, target_width, target_height);
        cv::Mat blurred   = filters::apply_blur(resized);
        cv::Mat clahe_out = filters::apply_clahe(blurred);
        cv::Mat sobel_out = filters::apply_sobel(clahe_out);
        cv::Mat sharpened = filters::apply_sharpen(sobel_out);

        // Write output
        writer.write(sharpened);
        frames_processed++;

        if (frames_processed % 50 == 0) {
            std::cout << "Processed " << frames_processed << " / " << total_frames << " frames...\n";
        }
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    cap.release();
    writer.release();

    std::cout << "Pipeline finished successfully.\n";
    std::cout << "Total Frames Processed: " << frames_processed << "\n";
    std::cout << "Total Execution Time:   " << elapsed.count() << " seconds\n";
    std::cout << "Average Throughput:     " << frames_processed / elapsed.count() << " FPS\n";

    return 0;
}