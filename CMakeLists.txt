cmake_minimum_required(VERSION 3.28)
project(AdaptiveVideoPipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Minimal executable
add_executable(video_pipeline src/main.cpp)
