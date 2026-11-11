#include "pipeline.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <thread>
#include <vector>
#include <memory>

#include "filters.hpp"
#include "frame_task.hpp"
#include "trace_manager.hpp"
#include "frame_reorder.hpp"
#include "bounded_queue.hpp"

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

void run_a3_static_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A3: Static Multicore Pipeline with " << num_workers << " workers...\n";
    TraceManager tracer("../traces/a3_trace.csv");
    
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    FrameReorder reorder(writer, tracer);
    
    // 1. Create a queue for each worker (bounded to hold max 10 frames at a time)
    std::vector<std::shared_ptr<BoundedQueue<FrameTask>>> queues;
    for (int i = 0; i < num_workers; ++i) {
        queues.push_back(std::make_shared<BoundedQueue<FrameTask>>(10));
    }

    // 2. Define the worker logic
    auto worker_func = [&](int worker_id) {
        std::string worker_name = "CPU_W" + std::to_string(worker_id);
        FrameTask task;
        
        // Keep popping frames until the queue is finished and empty
        while (queues[worker_id]->pop(task)) {
            tracer.recordEvent(task.frame_id, "resize", "start", worker_name);
            cv::Mat resized = filters::resize_frame(task.input_frame, 1280, 720);
            tracer.recordEvent(task.frame_id, "resize", "end", worker_name);
            
            tracer.recordEvent(task.frame_id, "blur", "start", worker_name);
            cv::Mat blurred = filters::apply_blur(resized);
            tracer.recordEvent(task.frame_id, "blur", "end", worker_name);
            
            tracer.recordEvent(task.frame_id, "clahe", "start", worker_name);
            cv::Mat clahe = filters::apply_clahe(blurred);
            tracer.recordEvent(task.frame_id, "clahe", "end", worker_name);
            
            tracer.recordEvent(task.frame_id, "sobel", "start", worker_name);
            cv::Mat sobel = filters::apply_sobel(clahe);
            tracer.recordEvent(task.frame_id, "sobel", "end", worker_name);
            
            tracer.recordEvent(task.frame_id, "sharpen", "start", worker_name);
            task.output_frame = filters::apply_sharpen(sobel);
            tracer.recordEvent(task.frame_id, "sharpen", "end", worker_name);
            
            // Send to reorder buffer
            reorder.push_and_write(task);
        }
    };

    // 3. Launch the threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker_func, i);
    }

    // 4. Main Thread reads frames and distributes statically
    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame;

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_MAIN");
        
        // Static Assignment Rule:
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // 5. Cleanup and wait for workers to finish
    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    cap.release(); 
    writer.release(); 
    tracer.flush();
    std::cout << "Pipeline A3 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}