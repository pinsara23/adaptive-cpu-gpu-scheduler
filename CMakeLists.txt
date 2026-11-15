cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(AdaptiveVideoPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- CUDA is optional: CPU algorithms (A0-A5) build without it ----
option(HPC_ENABLE_CUDA "Build GPU algorithms (A6+) when CUDA is available" ON)
include(CheckLanguage)
check_language(CUDA)
if(HPC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 3050 Laptop; pass -DCMAKE_CUDA_ARCHITECTURES=75 on RTX 2060
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HPC_HAS_CUDA ON)
    message(STATUS "CUDA found: building CPU + GPU algorithms")
else()
    set(HPC_HAS_CUDA OFF)
    message(STATUS "CUDA not found or disabled: building CPU algorithms only (A0-A5)")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp
    src/filters.cpp
    src/trace_manager.cpp
    src/pipeline.cpp
    src/frame_reorder.cpp
    src/cpu_pipeline.cpp
    src/cpu_topology.cpp
)
if(HPC_HAS_CUDA)
    list(APPEND SOURCES
        src/gpu_pipeline.cu
        src/gpu_blur.cu
        src/gpu_selftest.cpp      # uses GpuBlur5, so CUDA builds only
    )
endif()

add_executable(video_pipeline ${SOURCES})

if(HPC_HAS_CUDA)
    target_compile_definitions(video_pipeline PRIVATE HPC_HAS_CUDA)
endif()

target_compile_options(video_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>   # optional: warnings for host code in .cu
)
target_link_libraries(video_pipeline ${OpenCV_LIBS} Threads::Threads)