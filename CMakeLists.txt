cmake_minimum_required(VERSION 3.28)

# Default to an optimized build unless the user chose one explicitly.
# Must come BEFORE project() so the CUDA flags pick it up.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Native code for the RTX 3050 Laptop (compute capability 8.6).
# Change this on the RTX 2060 machine (Turing = 75).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(AdaptiveVideoPipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(video_pipeline
    src/main.cpp
    src/filters.cpp
    src/trace_manager.cpp
    src/pipeline.cpp
    src/frame_reorder.cpp
    src/cpu_pipeline.cpp
    src/gpu_pipeline.cu
    src/cpu_topology.cpp
)

target_compile_options(video_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_link_libraries(video_pipeline ${OpenCV_LIBS} Threads::Threads)