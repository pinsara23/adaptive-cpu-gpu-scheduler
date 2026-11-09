#include <iostream>
#include <string>
#include "pipeline.hpp"

int main(int argc, char** argv) {
    // Default to A0 if the user doesn't type an algorithm name
    std::string algo = "A0"; 
    if (argc > 1) {
        algo = argv[1];
    }

    std::string input_path = "../videos/input.mp4";
    std::string output_path = "../output/output_" + algo + ".mp4"; // Saves as output_A0.mp4, output_A1.mp4, etc.

    if (algo == "A0") {
        std::cout << "Executing Algorithm A0: Sequential Baseline...\n";
        run_sequential_pipeline(input_path, output_path);
    } 
    else {
        std::cerr << "Error: Unknown algorithm identifier '" << algo << "'\n";
        std::cerr << "Usage: ./video_pipeline [A0|A1|A2...]\n";
        return -1;
    }

    return 0;
}