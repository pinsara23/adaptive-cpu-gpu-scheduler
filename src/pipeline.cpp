#include "pipeline.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <thread>
#include <vector>
#include <memory>
#include <fstream>
#include <thread>
#include <vector>

#include "filters.hpp"
#include "frame_task.hpp"
#include "trace_manager.hpp"
#include "frame_reorder.hpp"
#include "bounded_queue.hpp"
#include "gpu_pipeline.hpp"

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

    auto start_time = chrono::steady_clock::now();

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

    cap.release();
    writer.release();

    auto end_time = std::chrono::steady_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

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

    auto start_time = chrono::steady_clock::now();

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


    cap.release();
    writer.release();

    auto end_time = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();


    cout << "Pipeline A1 finished successfully.\n";
    cout << "Total Frames: " << current_frame_id << "\n";
    cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a2_smt_pipeline(const std::string& input_path, const std::string& output_path) {
    std::cout << "Executing A2: Single-Core SMT (Hyper-Threading)...\n";
    
    // 1. Hardware-Aware Topology Read: Find the sibling thread for physical Core 0
    int sibling1 = 0;
    int sibling2 = 1; // Fallback
    std::ifstream sibling_file("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    if (sibling_file.is_open()) {
        std::string line;
        std::getline(sibling_file, line);
        size_t sep = line.find_first_of(",-");
        if (sep != std::string::npos) {
            sibling1 = std::stoi(line.substr(0, sep));
            sibling2 = std::stoi(line.substr(sep + 1));
        }
    }
    std::cout << "Hardware Topology: Physical Core 0 maps to logical threads CPU " << sibling1 << " and CPU " << sibling2 << "\n";

    TraceManager tracer("../traces/a2_trace.csv");
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    FrameReorder reorder(writer, tracer, total_frames);
    
    // 2. Create exactly 2 queues for the 2 SMT threads
    int num_workers = 2;
    std::vector<std::shared_ptr<BoundedQueue<FrameTask>>> queues;
    for (int i = 0; i < num_workers; ++i) {
        queues.push_back(std::make_shared<BoundedQueue<FrameTask>>(10));
    }

    // 3. Worker Logic with SMT Pinning
    auto worker_func = [&](int worker_id) {
        // Lock this thread specifically to the physical core's logical threads
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (worker_id == 0) {
            CPU_SET(sibling1, &cpuset);
        } else {
            CPU_SET(sibling2, &cpuset);
        }
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Warning: Failed to pin worker " << worker_id << "\n";
        }
        
        std::string worker_name = "CPU_SMT_" + std::to_string(worker_id);
        FrameTask task;
        
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
            
            reorder.push_and_write(std::move(task));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker_func, i);
    }

    // 4. Main Thread Dispatcher
    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame.clone(); // Ensure each task has its own copy of the frame

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_MAIN");
        
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    // NEW: Ensure buffer flushes and missing frames are logged before stopping the timer
    reorder.finalize();

    cap.release(); 
    writer.release(); 

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();


    std::cout << "Pipeline A2 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a3_static_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A3: Static Multicore Pipeline with " << num_workers << " workers...\n";
    TraceManager tracer("../traces/a3_trace.csv");
    
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    FrameReorder reorder(writer, tracer, total_frames);
    
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
            reorder.push_and_write(std::move(task));
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
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame.clone(); // Ensure each task has its own copy of the frame

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

    // NEW: Ensure buffer flushes and missing frames are logged before stopping the timer
    reorder.finalize();

    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << "Pipeline A3 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a4_dynamic_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers){
    
    // The Main Thread pushes decoded frames into this one central queue.

    // All workers (fast and slow) constantly check this queue.

    // The moment a worker finishes a frame, it immediately grabs the next available frame from the queue.

    std::cout << "Executing A4: Dynamic Multicore Pipeline with " << num_workers << " workers...\n";
    TraceManager tracer("../traces/a4_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    FrameReorder reorder(writer, tracer, total_frames);

    // 1. create 1 shared queue for all workers (bound to hold max 20 frames)
    auto shared_queue = std::make_shared<BoundedQueue<FrameTask>>(20);

    // 2. Define the worker logic (Dynamic polling)
    auto worker_func = [&](int worker_id){
        std::string worker_name = "CPU_W" + std::to_string(worker_id);
        FrameTask task;

        while(shared_queue->pop(task)){
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
            reorder.push_and_write(std::move(task));
        }
    };

    // 3. Launch the threads
    std::vector<std::thread> threads;
    for(int i=0; i<num_workers; i++){
        threads.emplace_back(worker_func, i);
    }
    
    // 4. Main thread reads frame and pushes to the single queue
    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame.clone(); // Ensure each task has its own copy of the frame

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_MAIN");

        shared_queue->push(task);

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // 5. Cleanup
    shared_queue->finish();
    for (auto& t : threads) t.join();

    // NEW: Ensure buffer flushes and missing frames are logged before stopping the timer
    reorder.finalize();

    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << "Pipeline A4 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a5_pinned_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A5: Thread-Pinned Multicore Pipeline with " << num_workers << " workers...\n";
    
    // 1. Discover dynamically available CPUs instead of hardcoding 20
    cpu_set_t available_cpus;
    sched_getaffinity(0, sizeof(cpu_set_t), &available_cpus);
    std::vector<int> online_cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &available_cpus)) {
            online_cpus.push_back(i);
        }
    }

    if (online_cpus.empty()) {
        online_cpus.push_back(0); // Safe fallback
    }

    std::cout << "Discovered " << online_cpus.size() << " online CPUs for pinning.\n";

    TraceManager tracer("../traces/a5_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    FrameReorder reorder(writer, tracer, total_frames);

    std::vector<std::shared_ptr<BoundedQueue<FrameTask>>> queues;
    for (int i = 0; i < num_workers; i++){
        queues.push_back(std::make_shared<BoundedQueue<FrameTask>>(10));
    }

    auto worker_func = [&](int worker_id){
        // PINNING LOGIC: Map to dynamically discovered online CPUs.
        // We offset by +1 so Worker 0 does not perfectly overlap with the Main Thread on CPU 0.
        int target_cpu = online_cpus[(worker_id + 1) % online_cpus.size()];
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target_cpu, &cpuset); 
        
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Warning: Failed to pin worker " << worker_id << " to CPU " << target_cpu << "\n";
        }

        std::string worker_name = "CPU_W" + std::to_string(worker_id) + "_PINNED";
        FrameTask task;

        while(queues[worker_id]->pop(task)){
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

            reorder.push_and_write(std::move(task));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back(worker_func, i);
    }

    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::steady_clock::now();

    // Pin the Main Thread to the first discovered core to isolate it from the workers
    cpu_set_t main_cpuset;
    CPU_ZERO(&main_cpuset);
    CPU_SET(online_cpus[0], &main_cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_cpuset) != 0) {
         std::cerr << "Warning: Failed to pin main thread to CPU " << online_cpus[0] << "\n";
    }

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame.clone(); // Deep copy preserved

        tracer.recordEvent(task.frame_id, "decode", "end", "CPU_MAIN_PINNED");
        
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    // NEW: Ensure buffer flushes and missing frames are logged before stopping the timer
    reorder.finalize();
    
    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << "Pipeline A5 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a6_gpu_pipeline(const std::string& input_path, const std::string& output_path) {
    std::cout << "Executing A6: Sequential CPU-GPU Heterogeneous Pipeline...\n";
    TraceManager tracer("../traces/a6_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    // FIX: Dynamically query available resources (Video Dimensions and FPS)
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           fps, cv::Size(frame_width, frame_height), false);

    // FIX: Allocate VRAM dynamically based on the specific input video
    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    allocate_device_memory(&d_in, &d_out, frame_width, frame_height, 3);

    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");
        
        // Deep copy for memory safety
        cv::Mat input_frame = raw_frame.clone();

        // --- CPU STAGE 1: Resize ---
        tracer.recordEvent(current_frame_id, "resize", "start", "CPU_MAIN");
        cv::Mat resized = filters::resize_frame(input_frame, frame_width, frame_height);
        tracer.recordEvent(current_frame_id, "resize", "end", "CPU_MAIN");

        // --- GPU STAGE 2: Custom CUDA Blur ---
        // Dynamically size the OpenCV output matrix to match
        cv::Mat blurred(frame_height, frame_width, CV_8UC3); 
        launch_cuda_blur(resized.data, blurred.data, d_in, d_out, frame_width, frame_height, 3, tracer, current_frame_id);

        // --- CPU STAGES 3, 4, 5: CLAHE, Sobel, Sharpen ---
        tracer.recordEvent(current_frame_id, "clahe", "start", "CPU_MAIN");
        cv::Mat clahe = filters::apply_clahe(blurred);
        tracer.recordEvent(current_frame_id, "clahe", "end", "CPU_MAIN");

        tracer.recordEvent(current_frame_id, "sobel", "start", "CPU_MAIN");
        cv::Mat sobel = filters::apply_sobel(clahe);
        tracer.recordEvent(current_frame_id, "sobel", "end", "CPU_MAIN");

        tracer.recordEvent(current_frame_id, "sharpen", "start", "CPU_MAIN");
        cv::Mat sharp = filters::apply_sharpen(sobel);
        tracer.recordEvent(current_frame_id, "sharpen", "end", "CPU_MAIN");

        // --- CPU STAGE 6: Encode ---
        tracer.recordEvent(current_frame_id, "encode", "start", "CPU_MAIN");
        writer.write(sharp);
        tracer.recordEvent(current_frame_id, "encode", "end", "CPU_MAIN");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Processed " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Clean up VRAM!
    free_device_memory(d_in, d_out);
    
    cap.release(); 
    writer.release(); 
    tracer.flush();
    
    std::cout << "Pipeline A6 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a7_concurrent_pipeline(const std::string& input_path, const std::string& output_path) {
    std::cout << "Executing A7: Concurrent CPU-GPU Pipeline...\n";
    TraceManager tracer("../traces/a7_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Queues: Capacity of 30 frames each to enforce backpressure
    BoundedQueue<FrameTask> cpu_to_gpu_queue(30);
    BoundedQueue<FrameTask> gpu_to_cpu_queue(30);

    auto start_time = std::chrono::steady_clock::now();

    // --- THREAD 3: Writer Thread (CPU Post-Processing & I/O) ---
    std::thread writer_thread([&]() {
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               fps, cv::Size(frame_width, frame_height), false);
        
        while (true) {
            FrameTask task;
            gpu_to_cpu_queue.pop(task);
            
            if (task.frame_id == -1) break; // Poison pill terminates thread

            tracer.recordEvent(task.frame_id, "clahe", "start", "CPU_WRITER");
            cv::Mat clahe = filters::apply_clahe(task.output_frame);
            tracer.recordEvent(task.frame_id, "clahe", "end", "CPU_WRITER");

            tracer.recordEvent(task.frame_id, "sobel", "start", "CPU_WRITER");
            cv::Mat sobel = filters::apply_sobel(clahe);
            tracer.recordEvent(task.frame_id, "sobel", "end", "CPU_WRITER");

            tracer.recordEvent(task.frame_id, "sharpen", "start", "CPU_WRITER");
            cv::Mat sharp = filters::apply_sharpen(sobel);
            tracer.recordEvent(task.frame_id, "sharpen", "end", "CPU_WRITER");

            tracer.recordEvent(task.frame_id, "encode", "start", "CPU_WRITER");
            writer.write(sharp);
            tracer.recordEvent(task.frame_id, "encode", "end", "CPU_WRITER");
        }
        writer.release();
    });

    // --- THREAD 2: GPU Worker Thread (PCIe & CUDA Compute) ---
    std::thread gpu_thread([&]() {
        unsigned char* d_in = nullptr;
        unsigned char* d_out = nullptr;
        allocate_device_memory(&d_in, &d_out, frame_width, frame_height, 3); // Pre-allocate VRAM

        while (true) {
            FrameTask task;
            cpu_to_gpu_queue.pop(task);
            
            if (task.frame_id == -1) {
                // Forward poison pill to Writer Thread and terminate
                gpu_to_cpu_queue.push(std::move(task));
                break;
            }

            cv::Mat blurred(frame_height, frame_width, CV_8UC3); 
            launch_cuda_blur(task.input_frame.data, blurred.data, d_in, d_out, frame_width, frame_height, 3, tracer, task.frame_id);

            // Assign the blurred frame to output_frame, and explicitly free input_frame to save RAM!
            task.output_frame = std::move(blurred);
            task.input_frame.release(); 

            gpu_to_cpu_queue.push(std::move(task)); // Zero-copy move to next queue
        }
        free_device_memory(d_in, d_out);
    });

    // --- THREAD 1: Main Reader Thread (CPU Pre-Processing) ---
    int current_frame_id = 0;
    cv::Mat raw_frame;
    
    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_READER");
        cv::Mat input_mat = raw_frame.clone(); // Deep copy once to prevent OpenCV data race

        tracer.recordEvent(current_frame_id, "resize", "start", "CPU_READER");
        cv::Mat resized = filters::resize_frame(input_mat, frame_width, frame_height);
        tracer.recordEvent(current_frame_id, "resize", "end", "CPU_READER");

        // Construct the FrameTask exactly according to your struct
        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = std::move(resized);

        cpu_to_gpu_queue.push(std::move(task)); // Push to GPU queue

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Read " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // Send Poison Pill to flush the queues and cleanly shutdown
    FrameTask poison_pill;
    poison_pill.frame_id = -1;
    cpu_to_gpu_queue.push(std::move(poison_pill));

    // Wait for workers to finish
    gpu_thread.join();
    writer_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    cap.release();
    tracer.flush();
    
    std::cout << "Pipeline A7 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a8_hybrid_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A8: Hybrid CPU-GPU Asynchronous Pipeline (" << num_workers << " workers)...\n";
    TraceManager tracer("../traces/a8_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) return;

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // FIX: Inline CPU topology discovery from A5
    cpu_set_t available_cpus;
    sched_getaffinity(0, sizeof(cpu_set_t), &available_cpus);
    std::vector<int> online_cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &available_cpus)) {
            online_cpus.push_back(i);
        }
    }
    if (online_cpus.empty()) {
        online_cpus.push_back(0); // Safe fallback
    }
    std::cout << "Discovered " << online_cpus.size() << " online CPUs for pinning.\n";

    // 3 Dedicated Queues
    BoundedQueue<FrameTask> cpu_to_gpu_queue(30);
    BoundedQueue<FrameTask> gpu_to_worker_queue(60);
    BoundedQueue<FrameTask> worker_to_writer_queue(60);

    auto start_time = std::chrono::steady_clock::now();

    // --- STAGE 4: Dedicated Writer Thread (CPU 1) ---
    std::thread writer_thread([&]() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(online_cpus[1 % online_cpus.size()], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               fps, cv::Size(frame_width, frame_height), false);
        
        FrameReorder reorder(writer, tracer, total_frames, 60); 

        while (true) {
            FrameTask task;
            worker_to_writer_queue.pop(task);
            
            if (task.frame_id == -1) break; // Poison pill

            reorder.push_and_write(std::move(task)); 
        }
        
        reorder.finalize(); 
        writer.release();
    });

    // --- STAGE 2: GPU Dispatch Thread (CPU 2) ---
    std::thread gpu_thread([&]() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(online_cpus[2 % online_cpus.size()], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        unsigned char* d_in = nullptr;
        unsigned char* d_out = nullptr;
        allocate_device_memory(&d_in, &d_out, frame_width, frame_height, 3);

        while (true) {
            FrameTask task;
            cpu_to_gpu_queue.pop(task);
            
            if (task.frame_id == -1) break; // Stop processing, do not poison workers yet

            cv::Mat blurred(frame_height, frame_width, CV_8UC3); 
            launch_cuda_blur(task.input_frame.data, blurred.data, d_in, d_out, frame_width, frame_height, 3, tracer, task.frame_id);

            task.output_frame = std::move(blurred);
            task.input_frame.release(); // Explicitly free RAM

            gpu_to_worker_queue.push(std::move(task));
        }
        free_device_memory(d_in, d_out);
    });

    // --- STAGE 3: CPU Worker Pool (CPUs 3+) ---
    std::vector<std::thread> workers;
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&, i]() {
            // Thread Pinning: Offset by 3 to reserve CPUs for Reader, Writer, and GPU dispatch
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            int target_cpu = online_cpus[(i + 3) % online_cpus.size()]; 
            CPU_SET(target_cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            
            std::string worker_id_str = "CPU_WORKER_" + std::to_string(i);

            while (true) {
                FrameTask task;
                gpu_to_worker_queue.pop(task);
                
                if (task.frame_id == -1) break;

                tracer.recordEvent(task.frame_id, "clahe", "start", worker_id_str);
                cv::Mat clahe = filters::apply_clahe(task.output_frame);
                tracer.recordEvent(task.frame_id, "clahe", "end", worker_id_str);

                tracer.recordEvent(task.frame_id, "sobel", "start", worker_id_str);
                cv::Mat sobel = filters::apply_sobel(clahe);
                tracer.recordEvent(task.frame_id, "sobel", "end", worker_id_str);

                tracer.recordEvent(task.frame_id, "sharpen", "start", worker_id_str);
                cv::Mat sharp = filters::apply_sharpen(sobel);
                tracer.recordEvent(task.frame_id, "sharpen", "end", worker_id_str);

                task.output_frame = std::move(sharp);
                worker_to_writer_queue.push(std::move(task));
            }
        });
    }

    // --- STAGE 1: Main Reader Thread (CPU 0) ---
    cpu_set_t main_cpuset;
    CPU_ZERO(&main_cpuset);
    CPU_SET(online_cpus[0], &main_cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_cpuset);

    int current_frame_id = 0;
    cv::Mat raw_frame;
    
    while (cap.read(raw_frame)) {
        if (raw_frame.empty()) break;
        
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_READER");
        cv::Mat input_mat = raw_frame.clone();

        tracer.recordEvent(current_frame_id, "resize", "start", "CPU_READER");
        cv::Mat resized = filters::resize_frame(input_mat, frame_width, frame_height);
        tracer.recordEvent(current_frame_id, "resize", "end", "CPU_READER");

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = std::move(resized);

        cpu_to_gpu_queue.push(std::move(task));

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Read " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // --- CASCADING SHUTDOWN SEQUENCE ---
    FrameTask gpu_poison;
    gpu_poison.frame_id = -1;
    cpu_to_gpu_queue.push(std::move(gpu_poison));
    gpu_thread.join();

    for (int i = 0; i < num_workers; ++i) {
        FrameTask worker_poison;
        worker_poison.frame_id = -1;
        gpu_to_worker_queue.push(std::move(worker_poison));
    }
    for (auto& worker : workers) {
        worker.join();
    }

    FrameTask writer_poison;
    writer_poison.frame_id = -1;
    worker_to_writer_queue.push(std::move(writer_poison));
    writer_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    cap.release();
    tracer.flush();
    
    std::cout << "Pipeline A8 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}