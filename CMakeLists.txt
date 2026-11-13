cmake_minimum_required(VERSION 3.28)

# 1. Enable BOTH C++ and CUDA languages
project(AdaptiveVideoPipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Set CUDA to use the same C++17 standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

# 3. Add the new CUDA (.cu) file to the executable
add_executable(video_pipeline 
    src/main.cpp 
    src/filters.cpp 
    src/trace_manager.cpp 
    src/pipeline.cpp 
    src/frame_reorder.cpp 
    src/gpu_pipeline.cu
)

target_link_libraries(video_pipeline ${OpenCV_LIBS})