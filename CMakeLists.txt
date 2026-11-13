cmake_minimum_required(VERSION 3.28)
# We must explicitly tell CMake to load both the C++ (CXX) and CUDA compilers
project(AdaptiveVideoPipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- NEW: CUDA Configuration for RTX 3050 Laptop GPU (Ampere) ---
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86) 

# Tell CMake to create a JSON file mapping all include paths for the IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include our custom headers and OpenCV headers
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Add our source files
add_executable(video_pipeline 
    src/main.cpp 
    src/filters.cpp 
    src/trace_manager.cpp 
    src/pipeline.cpp 
    src/frame_reorder.cpp
    src/gpu_pipeline.cu    
)

# Link OpenCV libraries to our executable
target_link_libraries(video_pipeline ${OpenCV_LIBS})