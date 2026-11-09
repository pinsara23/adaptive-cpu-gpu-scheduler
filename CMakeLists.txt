cmake_minimum_required(VERSION 3.28)
project(AdaptiveVideoPipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake to create a JSON file mapping all include paths for the IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include our custom headers and OpenCV headers
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Add our source files
add_executable(video_pipeline src/main.cpp src/filters.cpp)

# Link OpenCV libraries to our executable
target_link_libraries(video_pipeline ${OpenCV_LIBS})