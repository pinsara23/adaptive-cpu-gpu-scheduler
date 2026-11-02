# Adaptive CPU–GPU Scheduler for Parallel Video Processing

An architecture- and resource-aware adaptive scheduling framework for optimizing parallel video-processing pipelines on a single heterogeneous computing system.

## Overview

This project explores adaptive workload scheduling across CPU and GPU resources to improve the performance and resource utilization of video-processing pipelines.

Rather than relying on a fixed execution strategy, the proposed framework monitors system performance, estimates processing costs, and dynamically distributes workloads between CPU and GPU resources.

The system focuses on **pipeline parallelism, heterogeneous computing, GPU resource optimization, and adaptive workload scheduling**.

The project is developed as part of a High Performance Computing (HPC) academic project.

## Project Objectives

- Design and implement a parallel video-processing pipeline.
- Investigate CPU multicore and GPU parallel execution.
- Develop an adaptive CPU–GPU workload scheduling algorithm.
- Implement GPU resource auto-tuning.
- Explore concurrent CPU and GPU execution.
- Monitor CPU, GPU, memory, and data-transfer utilization.
- Evaluate different scheduling strategies using performance benchmarks.
- Identify bottlenecks and optimize overall pipeline performance.

## System Architecture

The proposed system consists of the following major components:

1. **Video Analyzer:** Extracts and validates video properties such as resolution, frame rate, duration, and frame count.
2. **Video Decoder:** Decodes input video into frames.
3. **Frame Queue Manager:** Manages frame batches, queues, and frame IDs.
4. **Adaptive CPU–GPU Scheduler (A10):** Determines how processing workloads should be distributed between CPU and GPU.
5. **GPU Resource Auto-Tuner (A9):** Evaluates GPU execution configurations such as CUDA streams, buffers, and frames in flight.
6. **CPU Processing Workers:** Execute assigned processing tasks using CPU parallelism.
7. **GPU Processing Workers:** Execute assigned workloads using CUDA.
8. **Performance Monitor:** Collects execution times and hardware resource utilization.
9. **Frame Reordering and Video Encoder:** Restores the correct frame order and produces the output video.

## Video Processing Pipeline

The initial video-processing workload consists of the following stages:

```text
Input Video
    |
    v
Video Decoding
    |
    v
Frame Queue Manager
    |
    v
Parallel Processing
    |
    +----------------------+
    |                      |
    v                      v
 CPU Workers           GPU Workers
    |                      |
    +----------+-----------+
               |
               v
       Frame Reordering
               |
               v
         Video Encoding
               |
               v
         Output Video
```

### Processing Stages

The planned image-processing operations are:

1. Frame decoding
2. Image resizing
3. Gaussian blur
4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Sobel edge detection
6. Image sharpening
7. Video encoding

Decoding, frame management, and encoding initially remain CPU-managed, while A10 focuses on adaptive distribution of image-processing workloads.

## Core Algorithms

### A9 — GPU Resource Auto-Tuner

A9 investigates suitable GPU execution configurations to improve GPU utilization and processing performance.

Parameters include:

- CUDA stream count
- Buffer count
- Frames in flight
- Batch size
- Queue capacity
- CPU–GPU execution overlap

The tuner evaluates feasible configurations while considering GPU memory limitations and execution overhead.

The objective is to identify efficient GPU configurations rather than assuming that increasing the number of CUDA streams always improves performance.

### A10 — Adaptive CPU–GPU Workload Scheduler

A10 dynamically distributes processing workloads across CPU and GPU resources.

The scheduler considers:

- CPU processing time
- GPU processing time
- CPU–GPU data-transfer overhead
- Synchronization overhead
- CPU and GPU utilization
- Queue conditions
- Available memory and GPU resources

The initial implementation uses performance measurements and cost estimation rather than machine learning.

#### Initial Calibration

The scheduler performs an initial calibration to estimate CPU and GPU processing performance.

#### Adaptive Cost Estimation

Execution-time estimates are updated using an Exponential Moving Average (EMA):

\[
E_{new} = \alpha T_{actual} + (1-\alpha)E_{old}
\]

Where:

- `E_new` — Updated execution-time estimate
- `T_actual` — Latest measured execution time
- `E_old` — Previous execution-time estimate
- `α` — Weight assigned to the latest measurement

#### Scheduling Strategies

A10 evaluates three execution strategies:

- **CPU-only:** Processing workloads execute on CPU resources.
- **GPU-only:** Processing workloads execute on GPU resources, with CPU support for decoding, transfers, and encoding.
- **Hybrid:** CPU and GPU process independent workloads concurrently.

The scheduler uses estimated end-to-end execution costs to determine a suitable workload distribution.

### A11 — Integrated Adaptive Scheduling Framework

A11 integrates A9 and A10 into a unified scheduling framework.

It combines:

- Adaptive CPU–GPU workload distribution
- GPU resource auto-tuning
- Performance monitoring
- Queue management
- Cost estimation
- Resource-aware execution decisions

## Parallel Execution Strategies

The framework will evaluate several execution approaches:

| Strategy | Description |
|---|---|
| Sequential CPU | Processes the pipeline sequentially |
| Multithreaded CPU | Uses multiple CPU threads |
| Fixed GPU | Uses a predefined GPU configuration |
| Fixed Hybrid | Uses a predefined CPU–GPU workload distribution |
| Adaptive CPU–GPU | Dynamically adjusts workload distribution |
| Adaptive Integrated | Combines A9 and A10 |

Both pipeline parallelism and data parallelism will be investigated.

## Performance Evaluation

The framework will be evaluated using controlled experiments.

### Performance Metrics

| Metric | Purpose |
|---|---|
| Total execution time | Overall processing performance |
| Throughput (FPS) | Frames processed per second |
| Stage execution time | Identifies pipeline bottlenecks |
| CPU utilization | Measures CPU resource usage |
| GPU utilization | Measures GPU resource usage |
| RAM consumption | Measures system memory usage |
| VRAM consumption | Measures GPU memory usage |
| Data-transfer time | Evaluates CPU–GPU transfer overhead |
| Synchronization overhead | Measures coordination costs |
| Scheduling overhead | Evaluates adaptive scheduling costs |

### Benchmarking Methodology

- Use identical input videos across execution strategies.
- Separate initial calibration and warm-up from measured runs.
- Repeat experiments to reduce measurement variability.
- Verify output correctness and frame ordering.
- Record hardware utilization and execution metrics.
- Compare fixed and adaptive scheduling approaches.

No performance improvements will be assumed before experimental validation.

## Development Environment

Initial development hardware:

| Component | Specification |
|---|---|
| CPU | Intel Core i7-12700H |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU Memory | 4 GB |
| System RAM | 16 GB |
| Operating System | Ubuntu Linux |

Additional testing may be performed on university systems equipped with NVIDIA RTX 2060 GPUs.

### Technologies

The planned technology stack includes:

- C++
- CUDA C++
- OpenMP
- OpenCV
- FFmpeg / FFprobe
- CMake
- NVIDIA Nsight Systems
- Linux performance-monitoring tools

## Project Development Roadmap

The project is divided into two academic semesters.

### Semester 1 — Algorithm Development and Initial Integration

- [ ] Set up the development environment.
- [ ] Implement the video-processing pipeline.
- [ ] Develop sequential and multithreaded CPU baselines.
- [ ] Implement GPU processing baselines.
- [ ] Develop initial A9 GPU resource auto-tuning.
- [ ] Develop initial A10 adaptive CPU–GPU scheduling.
- [ ] Perform initial integration and testing.

### Semester 2 — Evaluation and Optimization

- [ ] Integrate A9 and A10 into A11.
- [ ] Implement detailed performance monitoring.
- [ ] Conduct controlled benchmarking experiments.
- [ ] Evaluate resource utilization.
- [ ] Analyze scheduling overhead and bottlenecks.
- [ ] Refine scheduling and tuning strategies.
- [ ] Complete final testing, documentation, and demonstration.

## Repository Structure

The proposed repository structure is:

```text
adaptive-cpu-gpu-scheduler/
│
├── README.md
│
├── docs/
│   ├── architecture/
│   ├── algorithms/
│   ├── methodology/
│   └── reports/
│
├── src/
│   ├── decoder/
│   ├── pipeline/
│   ├── scheduler/
│   │   ├── a10/
│   │   └── a11/
│   ├── gpu_tuner/
│   │   └── a9/
│   ├── cpu/
│   ├── gpu/
│   ├── encoder/
│   └── monitoring/
│
├── benchmarks/
│   ├── cpu/
│   ├── gpu/
│   ├── hybrid/
│   └── results/
│
├── tests/
│
├── scripts/
│
├── configs/
│
└── CMakeLists.txt
```

This is the planned structure and may evolve during implementation

## Expected Outcomes

The project aims to deliver:

- A parallel video-processing framework.
- A GPU resource auto-tuning algorithm.
- An adaptive CPU–GPU workload scheduling algorithm.
- An integrated scheduling framework.
- A performance evaluation comparing multiple execution strategies.
- An analysis of resource utilization, execution overhead, and scheduling effectiveness.

## References

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [OpenMP Specifications](https://www.openmp.org/specifications/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

## Project Status

**Status:** Under Development

This project is being developed incrementally, with algorithm implementation, integration, benchmarking, and optimization planned across two academic semesters.

---

**Adaptive CPU–GPU Scheduling | Parallel Computing | High Performance Computing**