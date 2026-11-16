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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include "filters.hpp"
#include "frame_task.hpp"
#include "trace_manager.hpp"
#include "frame_reorder.hpp"
#include "bounded_queue.hpp"
#include "gpu_pipeline.hpp"
#include "cpu_pipeline.hpp"
#include "config.hpp"
#include "cpu_topology.hpp"
#ifdef HPC_HAS_CUDA
#include "gpu_blur.hpp"
#include "gpu_slot_pool.hpp"
#endif

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

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    int current_frame_id = 0;
    cv::Mat raw_frame;

    auto start_time = chrono::steady_clock::now();

    while (true) {
        
        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");   
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame;

        task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                              "CPU_MAIN", target_width, target_height);
        
        tracer.recordEvent(task.frame_id, "encode", "start", "CPU_MAIN");
        if (!config::no_encode) writer.write(task.output_frame);
        tracer.recordEvent(task.frame_id, "encode", "end", "CPU_MAIN");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Processed " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    auto loop_end = std::chrono::steady_clock::now();   // all frames processed and handed to the writer
    cap.release();
    writer.release();                                   // finalizes the mp4
    auto end_time = std::chrono::steady_clock::now();
    chrono::duration<double> loop_elapsed = loop_end - start_time;
    chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    if (total_frames > 0 && current_frame_id != total_frames) {
        std::cerr << "WARNING: decoded " << current_frame_id
                  << " frames but container reports " << total_frames << "\n";
    }

    cout << "Pipeline A0 finished successfully.\n";
    cout << "Total Frames: " << current_frame_id << "\n";
    cout << "Frames Emitted: " << current_frame_id << "\n";
    cout << "Loop Time:    " << loop_elapsed.count() << " seconds\n";
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
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    int current_frame_id = 0;
    cv::Mat raw_frame;

    auto start_time = chrono::steady_clock::now();

    // 3. Process video (locked to 1 core)

    while (true) {

        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_0");   
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_0");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        task.input_frame = raw_frame;

        task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                              "CPU_0", target_width, target_height);

        tracer.recordEvent(task.frame_id, "encode", "start", "CPU_0");
        if (!config::no_encode) writer.write(task.output_frame);
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

    if (total_frames > 0 && current_frame_id != total_frames) {
        std::cerr << "WARNING: decoded " << current_frame_id
                  << " frames but container reports " << total_frames << "\n";
    }


    cout << "Pipeline A1 finished successfully.\n";
    cout << "Total Frames: " << current_frame_id << "\n";
    cout << "Frames Emitted: " << current_frame_id << "\n";
    cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a2_smt_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {

    std::cout << "Executing A2: Single-Core SMT with " << num_workers << " worker(s)...\n";

        // Workers share ONE physical core (core 0). The main/decoder thread runs on a different
        // physical core, so A2(1) vs A2(2) differs only in SMT usage.
        std::vector<int> sib = cpu_siblings(0);   // e.g. {0, 1}
        if (num_workers < 1) num_workers = 1;
        if (num_workers > static_cast<int>(sib.size())) {
            std::cerr << "Warning: core 0 has only " << sib.size()
                      << " logical CPU(s); using that many workers\n";
            num_workers = static_cast<int>(sib.size());
        }

    // Pin the main thread BEFORE opening the video: FFmpeg's decoder threads inherit it.
    int main_cpu = -1;
    for (int c : cpu_assignment_order()) {
        if (std::find(sib.begin(), sib.end(), c) == sib.end()) { main_cpu = c; break; }
    }
    if (main_cpu >= 0) {
        cpu_set_t ms;
        CPU_ZERO(&ms);
        CPU_SET(main_cpu, &ms);
        if (pthread_setaffinity_np(pthread_self(), sizeof(ms), &ms) != 0) {
            std::cerr << "Warning: failed to pin main thread to CPU " << main_cpu << "\n";
        }
    }
    std::cout << "A2 pinning: main/decoder -> CPU " << main_cpu << ", workers ->";
    for (int i = 0; i < num_workers; ++i) std::cout << " " << sib[i];
    std::cout << "\n";

    TraceManager tracer("../traces/a2_trace.csv");
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    FrameReorder reorder(writer, tracer, total_frames);
    
    std::vector<std::shared_ptr<BoundedQueue<FrameTask>>> queues;
    for (int i = 0; i < num_workers; ++i) {
        queues.push_back(std::make_shared<BoundedQueue<FrameTask>>(10));
    }

    // 3. Worker Logic with SMT Pinning
    auto worker_func = [&](int worker_id) {
        
        // Lock this thread specifically to the physical core's logical threads
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(sib[worker_id], &cpuset);
        
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Warning: Failed to pin worker " << worker_id << "\n";
        }
        
        std::string worker_name = "CPU_SMT_" + std::to_string(worker_id);
        FrameTask task;
        
        while (queues[worker_id]->pop(task)) {      // A4: shared_queue->pop(task)
            task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                                  worker_name,
                                                  config::TARGET_WIDTH, config::TARGET_HEIGHT);
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

    while (true) {
        
        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");    
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);

        tracer.recordEvent(task.frame_id, "copy", "start", "CPU_MAIN");        
        task.input_frame = raw_frame.clone();
        tracer.recordEvent(task.frame_id, "copy", "end", "CPU_MAIN");          

        tracer.recordEvent(task.frame_id, "enqueue", "start", "CPU_MAIN");
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        tracer.recordEvent(task.frame_id, "enqueue", "end", "CPU_MAIN");
        
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    // Ensure buffer flushes and missing frames are logged before stopping the timer
    bool ok = reorder.finalize(current_frame_id);

    cap.release(); 
    writer.release(); 

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();



    std::cout << (ok ? "Pipeline A2 finished successfully.\n" : "Pipeline A2 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a3_static_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A3: Static Multicore Pipeline with " << num_workers << " workers...\n";
    TraceManager tracer("../traces/a3_trace.csv");
    
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

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
        while (queues[worker_id]->pop(task)) {      // A4: shared_queue->pop(task)
            task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                                  worker_name,
                                                  config::TARGET_WIDTH, config::TARGET_HEIGHT);
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

    while (true) {

        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");    
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);

        tracer.recordEvent(task.frame_id, "copy", "start", "CPU_MAIN");        
        task.input_frame = raw_frame.clone();
        tracer.recordEvent(task.frame_id, "copy", "end", "CPU_MAIN");          

        tracer.recordEvent(task.frame_id, "enqueue", "start", "CPU_MAIN");
        // Static Assignment Rule:
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        tracer.recordEvent(task.frame_id, "enqueue", "end", "CPU_MAIN");
        
        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // 5. Cleanup and wait for workers to finish
    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    // NEW: Ensure buffer flushes and missing frames are logged before stopping the timer
    bool ok = reorder.finalize(current_frame_id);

    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << (ok ? "Pipeline A3 finished successfully.\n" : "Pipeline A3 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a4_dynamic_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers){
    
    // The Main Thread pushes decoded frames into this one central queue.

    // All workers (fast and slow) constantly check this queue.

    // The moment a worker finishes a frame, it immediately grabs the next available frame from the queue.

    std::cout << "Executing A4: Dynamic Multicore Pipeline with " << num_workers << " workers...\n";
    TraceManager tracer("../traces/a4_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    FrameReorder reorder(writer, tracer, total_frames);

    // 1. create 1 shared queue for all workers (bound to hold max 20 frames)
    auto shared_queue = std::make_shared<BoundedQueue<FrameTask>>(20);

    // 2. Define the worker logic (Dynamic polling)
    auto worker_func = [&](int worker_id){
        std::string worker_name = "CPU_W" + std::to_string(worker_id);
        FrameTask task;

        while (shared_queue->pop(task)) {      
            task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                                  worker_name,
                                                  config::TARGET_WIDTH, config::TARGET_HEIGHT);
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

    while (true) {
        
        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");    
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);

        tracer.recordEvent(task.frame_id, "copy", "start", "CPU_MAIN");        
        task.input_frame = raw_frame.clone();
        tracer.recordEvent(task.frame_id, "copy", "end", "CPU_MAIN");          

        tracer.recordEvent(task.frame_id, "enqueue", "start", "CPU_MAIN");     
        shared_queue->push(task);
        tracer.recordEvent(task.frame_id, "enqueue", "end", "CPU_MAIN");       

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    // 5. Cleanup
    shared_queue->finish();
    for (auto& t : threads) t.join();

    bool ok = reorder.finalize(current_frame_id);
    

    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << (ok ? "Pipeline A4 finished successfully.\n" : "Pipeline A4 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

void run_a5_pinned_multicore_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A5: Thread-Pinned Multicore Pipeline with " << num_workers << " workers...\n";
    
    std::vector<int> online_cpus = cpu_assignment_order();
    
    std::cout << "Discovered " << online_cpus.size() << " online CPUs for pinning.\n";

    std::cout << "A5 pinning: main -> CPU " << online_cpus[0] << ", workers ->";
    for (int i = 0; i < num_workers; ++i) std::cout << " " << online_cpus[(i + 1) % online_cpus.size()];
    std::cout << "\n";

    TraceManager tracer("../traces/a5_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                           cap.get(cv::CAP_PROP_FPS), cv::Size(1280, 720), false);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

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

        while (queues[worker_id]->pop(task)) {      // A4: shared_queue->pop(task)
            task.output_frame = process_frame_cpu(task.input_frame, task.frame_id, tracer,
                                                  worker_name,
                                                  config::TARGET_WIDTH, config::TARGET_HEIGHT);
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

    while (true) {

        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN_PINNED");   
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN_PINNED");     

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);

        tracer.recordEvent(task.frame_id, "copy", "start", "CPU_MAIN_PINNED");        
        task.input_frame = raw_frame.clone();
        tracer.recordEvent(task.frame_id, "copy", "end", "CPU_MAIN_PINNED");          
        
        tracer.recordEvent(task.frame_id, "enqueue", "start", "CPU_MAIN_PINNED");     
        int target_worker = current_frame_id % num_workers;
        queues[target_worker]->push(task);
        tracer.recordEvent(task.frame_id, "enqueue", "end", "CPU_MAIN_PINNED");       

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    for (auto& q : queues) q->finish();
    for (auto& t : threads) t.join();

    bool ok = reorder.finalize(current_frame_id);
    
    cap.release(); 
    writer.release(); 
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    std::cout << (ok ? "Pipeline A5 finished successfully.\n" : "Pipeline A5 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
}

#ifdef HPC_HAS_CUDA
namespace {
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}
}  // namespace

// A6: Single-stream GPU baseline.
// Same stages, resolution, parameters, timing boundary and output format as A0; the
// blur runs on the GPU (GpuBlur5, bit-identical to cv::GaussianBlur). One thread, one
// CUDA stream, synchronous: the CPU waits while the GPU works (no CPU-GPU overlap).
void run_a6_gpu_pipeline(const std::string& input_path, const std::string& output_path) {
    std::cout << "Executing A6: Single-Stream GPU Pipeline (blur on GPU)...\n";
    TraceManager tracer("../traces/a6_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open input video at " << input_path << "\n";
        return;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int W = config::TARGET_WIDTH;
    const int H = config::TARGET_HEIGHT;

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(W, H), false);
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    // GPU setup, excluded from timing like the writer setup. Allocating the pinned and
    // device buffers also creates the CUDA context.
    const auto setup_t0 = std::chrono::steady_clock::now();
    GpuBlur5 gpu(W, H, 3);
    cv::Mat staged(H, W, CV_8UC3, gpu.host_in());    // resize writes here (pinned)
    cv::Mat blurred(H, W, CV_8UC3, gpu.host_out());  // GPU result lands here (pinned)
    staged.setTo(cv::Scalar::all(0));
    gpu.run_sync();   // warm-up: one-time costs (e.g. kernel loading) stay out of the timing
    const double setup_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_t0).count();

    struct FrameGpuTimes { int frame_id; GpuBlurTimings t; };
    std::vector<FrameGpuTimes> gpu_times;
    gpu_times.reserve(total_frames > 0 ? total_frames : 1024);

    int current_frame_id = 0;
    cv::Mat raw_frame;

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");

        tracer.recordEvent(current_frame_id, "resize", "start", "CPU_MAIN");
        filters::resize_into(raw_frame, staged, W, H);
        tracer.recordEvent(current_frame_id, "resize", "end", "CPU_MAIN");
        CV_Assert(staged.data == gpu.host_in());   // must not have reallocated

        tracer.recordEvent(current_frame_id, "blur", "start", "GPU_0");
        const GpuBlurTimings t = gpu.run_sync();
        tracer.recordEvent(current_frame_id, "blur", "end", "GPU_0");
        gpu_times.push_back({current_frame_id, t});

        // Reading the pinned output directly is safe only because A6 is sequential:
        // the next run_sync cannot overwrite it before these CPU stages finish.
        cv::Mat out = process_tail_cpu(blurred, current_frame_id, tracer, "CPU_MAIN");

        tracer.recordEvent(current_frame_id, "encode", "start", "CPU_MAIN");
        if (!config::no_encode) writer.write(out);
        tracer.recordEvent(current_frame_id, "encode", "end", "CPU_MAIN");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Processed " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    auto loop_end = std::chrono::steady_clock::now();
    cap.release();
    writer.release();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> loop_elapsed = loop_end - start_time;
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    // Per-frame GPU timings from CUDA events (written after timing stops).
    std::vector<double> h2d, kern, d2h, wall;
    std::ofstream csv("../traces/a6_gpu_timings.csv");
    csv << "frame_id,h2d_ms,kernel_ms,d2h_ms,wall_ms\n";
    for (const auto& g : gpu_times) {
        csv << g.frame_id << ',' << g.t.h2d_ms << ',' << g.t.kernel_ms << ','
            << g.t.d2h_ms << ',' << g.t.wall_ms << '\n';
        h2d.push_back(g.t.h2d_ms);
        kern.push_back(g.t.kernel_ms);
        d2h.push_back(g.t.d2h_ms);
        wall.push_back(g.t.wall_ms);
    }

    if (total_frames > 0 && current_frame_id != total_frames) {
        std::cerr << "WARNING: decoded " << current_frame_id
                  << " frames but container reports " << total_frames << "\n";
    }

    std::cout << "Pipeline A6 finished successfully.\n";
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << current_frame_id << "\n";
    std::cout << "Loop Time:    " << loop_elapsed.count() << " seconds\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
    std::cout << "GPU Setup Time: " << setup_s << " seconds (excluded from Total Time)\n";
    std::printf("GPU blur per frame (median, CUDA events): H2D %.3f ms, kernels %.3f ms, "
                "D2H %.3f ms, wall %.3f ms\n",
                median_of(h2d), median_of(kern), median_of(d2h), median_of(wall));
}

// A7: Fixed multi-stream GPU.
// Same structure as A4 (main thread decodes into one shared bounded queue, M workers,
// reorder buffer encodes in order) and the same stages; the blur runs on the GPU in
// one of S stream slots. Host buffers belong to a worker, device buffers to a slot.
void run_a7_multistream_pipeline(const std::string& input_path, const std::string& output_path,
                                 int num_workers, int num_slots) {
    std::cout << "Executing A7: Multi-Stream GPU Pipeline (" << num_workers << " workers, "
              << num_slots << " GPU slots)...\n";
    TraceManager tracer("../traces/a7_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int W = config::TARGET_WIDTH;
    const int H = config::TARGET_HEIGHT;
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           cap.get(cv::CAP_PROP_FPS), cv::Size(W, H), false);
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video for writing: " << output_path << "\n";
        return;
    }

    FrameReorder reorder(writer, tracer, total_frames);
    auto shared_queue = std::make_shared<BoundedQueue<FrameTask>>(20);

    // GPU setup (excluded from Total Time, reported separately): S slots (stream + device
    // buffers), one pinned input/output pair per worker, one warm-up frame per slot.
    const auto setup_t0 = std::chrono::steady_clock::now();
    GpuSlotPool pool(num_slots, W, H, 3);
    const size_t bytes = static_cast<size_t>(W) * H * 3;
    std::vector<std::unique_ptr<PinnedBuffer>> pin_in, pin_out;
    for (int w = 0; w < num_workers; ++w) {
        pin_in.push_back(std::make_unique<PinnedBuffer>(bytes));
        pin_out.push_back(std::make_unique<PinnedBuffer>(bytes));
    }
    std::memset(pin_in[0]->data(), 0, bytes);
    for (int s = 0; s < num_slots; ++s) pool.slot(s).run_sync(pin_in[0]->data(), pin_out[0]->data());
    const double setup_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_t0).count();

    std::vector<std::string> slot_names;
    for (int s = 0; s < num_slots; ++s) slot_names.push_back("GPU_S" + std::to_string(s));

    struct GpuRecord { int frame_id; int worker; int slot; double wait_ms; GpuBlurTimings t; };
    std::vector<std::vector<GpuRecord>> records(num_workers);   // one per worker, no locking

    auto worker_func = [&](int worker_id) {
        try {
            const std::string name = "CPU_W" + std::to_string(worker_id);
            cv::Mat staged(H, W, CV_8UC3, pin_in[worker_id]->data());    // this worker's pinned input
            cv::Mat blurred(H, W, CV_8UC3, pin_out[worker_id]->data());  // this worker's pinned output
            auto& my_records = records[worker_id];
            FrameTask task;

            while (shared_queue->pop(task)) {
                tracer.recordEvent(task.frame_id, "resize", "start", name);
                filters::resize_into(task.input_frame, staged, W, H);
                tracer.recordEvent(task.frame_id, "resize", "end", name);
                CV_Assert(staged.data == pin_in[worker_id]->data());
                task.input_frame.release();

                tracer.recordEvent(task.frame_id, "gpu_wait", "start", name);
                const auto w0 = std::chrono::steady_clock::now();
                const int s = pool.acquire();
                const double wait_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - w0).count();
                tracer.recordEvent(task.frame_id, "gpu_wait", "end", name);

                tracer.recordEvent(task.frame_id, "blur", "start", slot_names[s]);
                const GpuBlurTimings t = pool.slot(s).run_sync(staged.data, blurred.data);
                pool.release(s);
                tracer.recordEvent(task.frame_id, "blur", "end", slot_names[s]);
                my_records.push_back({task.frame_id, worker_id, s, wait_ms, t});

                // `blurred` is reused only by this worker's next frame, after this tail
                // finishes; the tail writes new images, so output_frame never aliases it.
                task.output_frame = process_tail_cpu(blurred, task.frame_id, tracer, name);
                reorder.push_and_write(std::move(task));
            }
        } catch (const std::exception& e) {
            // A lost frame would stall the reorder buffer, so a worker error is fatal.
            std::fprintf(stderr, "A7 worker %d failed: %s\n", worker_id, e.what());
            std::_Exit(2);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; i++) threads.emplace_back(worker_func, i);

    int current_frame_id = 0;
    cv::Mat raw_frame;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        tracer.recordEvent(current_frame_id, "decode", "start", "CPU_MAIN");
        if (!cap.read(raw_frame) || raw_frame.empty()) break;
        tracer.recordEvent(current_frame_id, "decode", "end", "CPU_MAIN");

        FrameTask task;
        task.frame_id = current_frame_id;
        task.video_timestamp = cap.get(cv::CAP_PROP_POS_MSEC);

        tracer.recordEvent(task.frame_id, "copy", "start", "CPU_MAIN");
        task.input_frame = raw_frame.clone();
        tracer.recordEvent(task.frame_id, "copy", "end", "CPU_MAIN");

        tracer.recordEvent(task.frame_id, "enqueue", "start", "CPU_MAIN");
        shared_queue->push(task);
        tracer.recordEvent(task.frame_id, "enqueue", "end", "CPU_MAIN");

        current_frame_id++;
        if (current_frame_id % 50 == 0) {
            std::cout << "Decoded & Dispatched " << current_frame_id << " / " << total_frames << " frames...\n";
        }
    }

    shared_queue->finish();
    for (auto& t : threads) t.join();

    bool ok = reorder.finalize(current_frame_id);

    cap.release();
    writer.release();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    tracer.flush();

    // Per-frame GPU timings and slot waits (written after timing stops).
    std::vector<double> h2d, kern, d2h, wall, waits;
    std::ofstream csv("../traces/a7_gpu_timings.csv");
    csv << "frame_id,worker,slot,wait_ms,h2d_ms,kernel_ms,d2h_ms,wall_ms\n";
    for (const auto& per_worker : records) {
        for (const auto& r : per_worker) {
            csv << r.frame_id << ',' << r.worker << ',' << r.slot << ',' << r.wait_ms << ','
                << r.t.h2d_ms << ',' << r.t.kernel_ms << ',' << r.t.d2h_ms << ',' << r.t.wall_ms << '\n';
            h2d.push_back(r.t.h2d_ms); kern.push_back(r.t.kernel_ms);
            d2h.push_back(r.t.d2h_ms); wall.push_back(r.t.wall_ms);
            waits.push_back(r.wait_ms);
        }
    }
    double wait_mean = 0.0;
    for (double w : waits) wait_mean += w;
    if (!waits.empty()) wait_mean /= waits.size();

    std::cout << (ok ? "Pipeline A7 finished successfully.\n" : "Pipeline A7 finished WITH ERRORS.\n");
    std::cout << "Total Frames: " << current_frame_id << "\n";
    std::cout << "Frames Emitted: " << reorder.frames_emitted() << "\n";
    std::cout << "Total Time:   " << elapsed.count() << " seconds\n";
    std::cout << "GPU Setup Time: " << setup_s << " seconds (excluded from Total Time)\n";
    std::printf("GPU blur per frame (median, CUDA events): H2D %.3f ms, kernels %.3f ms, "
                "D2H %.3f ms, wall %.3f ms\n",
                median_of(h2d), median_of(kern), median_of(d2h), median_of(wall));
    std::printf("GPU slot wait per frame: mean %.3f ms, median %.3f ms (max frames on GPU: %d)\n",
                wait_mean, median_of(waits), std::min(num_workers, num_slots));
}

void run_a8_hybrid_pipeline(const std::string& input_path, const std::string& output_path, int num_workers) {
    std::cout << "Executing A8: Hybrid CPU-GPU Asynchronous Pipeline (" << num_workers << " workers)...\n";
    TraceManager tracer("../traces/a8_trace.csv");

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Error: Could not open input video at " << input_path << "\n"; return; }

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
#endif // HPC_HAS_CUDA
