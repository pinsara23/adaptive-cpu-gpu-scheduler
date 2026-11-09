#include "pipeline.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>

#include "filters.hpp"
#include "frame_task.hpp"
#include "trace_manager.hpp"

using namespace std; 

void run_sequential_pipeline(const string& input_path, const std::string& output_path) {

    TraceManager tracer("../traces/sequential_trace.csv");
    
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open input video at " << input_path << "\n";
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int target_width = 1280;
    int target_height = 720;

    cv::VideoWriter writer(output_path, 
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           fps, cv::Size(target_width, target_height), false);

    int current_frame_id = 0;
    cv::Mat raw_frame;

    auto start_time = chrono::high_resolution_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame;

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "resize", "start", "CPU_MAIN");
        cv::Mat resized = filters::resize_frame(task.input_frame, target_width, target_height);
        tracer.recordEvent(task.frame_id, "resize", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "blur", "start", "CPU_MAIN");
        cv::Mat blurred = filters::apply_blur(resized);
        tracer.recordEvent(task.frame_id, "blur", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "clahe", "start", "CPU_MAIN");
        cv::Mat clahe_out = filters::apply_clahe(blurred);
        tracer.recordEvent(task.frame_id, "clahe", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "sobel", "start", "CPU_MAIN");
        cv::Mat sobel_out = filters::apply_sobel(clahe_out);
        tracer.recordEvent(task.frame_id, "sobel", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "sharpen", "start", "CPU_MAIN");
        task.output_frame = filters::apply_sharpen(sobel_out);
        tracer.recordEvent(task.frame_id, "sharpen", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "encode", "start", "CPU_MAIN");
        writer.write(task.output_frame);
        tracer.recordEvent(task.frame_id, "encode", "end", "CPU_MAIN");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Processed " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cap.release();
    writer.release();
    tracer.flush();

    cout << "Pipeline A0 finished successfully.\n";
    cout << "Total Frames: " << current_frame_id << "\n";
    cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a1_single_core_pipeline(const std::string& input_path, const std::string& output_path) {

    cout<<"Looking pipeline to CPU core 0 (P-Core)... \n";

    // 1. pin the specific thread to core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Pin to core 0

    pthread_t current_thread = pthread_self();
    int pin_result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (pin_result != 0) {
        cerr<<"Warning: Could not pin thread to Core 0. \n";
    }

    // 2. Initialize tracing and video streams
    TraceManager tracer("../traces/a1_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr<<"Error: Could not open input video! \n";
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int target_width = 1280;
    int target_height = 720;

    cv::VideoWriter writer(output_path, 
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           fps, cv::Size(target_width, target_height), false);

    int current_frame_id = 0;
    cv::Mat raw_frame;

    auto start_time = chrono::high_resolution_clock::now();

    // 3. Process video (locked to 1 core)

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame;

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "resize", "start", "CPU_0");
        cv::Mat resized = filters::resize_frame(task.input_frame, target_width, target_height);
        tracer.recordEvent(task.frame_id, "resize", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "blur", "start", "CPU_0");
        cv::Mat blurred = filters::apply_blur(resized);
        tracer.recordEvent(task.frame_id, "blur", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "clahe", "start", "CPU_0");
        cv::Mat clahe_out = filters::apply_clahe(blurred);
        tracer.recordEvent(task.frame_id, "clahe", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "sobel", "start", "CPU_0");
        cv::Mat sobel_out = filters::apply_sobel(clahe_out);
        tracer.recordEvent(task.frame_id, "sobel", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "sharpen", "start", "CPU_0");
        task.output_frame = filters::apply_sharpen(sobel_out);
        tracer.recordEvent(task.frame_id, "sharpen", "end", "CPU_0");

        tracer.recordEvent(task.frame_id, "encode", "start", "CPU_0");
        writer.write(task.output_frame);
        tracer.recordEvent(task.frame_id, "encode", "end", "CPU_0");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            cout<<"Processed "<<current_frame_id<<" / "<<total_frames<<" frames...\n";
        }

    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cap.release();
    writer.release();
    tracer.flush();

    cout << "Pipeline A1 finished successfully.\n";
    cout << "Total Frames: " << current_frame_id << "\n";
    cout << "Total Time:   " << elapsed.count() << " seconds\n";
}