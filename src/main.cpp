#include <iostream>
#include <string>
#include "pipeline.hpp"

using namespace std;

int main(int argc, char** argv) {
    // Default to A0 if the user doesn't type an algorithm name
    string algo = "A0"; 
    if (argc > 1) {
        algo = argv[1];
    }

    string input_path = "../videos/input.mp4";
    string output_path = "../output/output_" + algo + ".mp4"; // Saves as output_A0.mp4, output_A1.mp4, etc.

    if (algo == "A0") {
        cout << "Executing Algorithm A0: Sequential Baseline...\n";
        run_sequential_pipeline(input_path, output_path);
    } 
    else if (algo == "A1"){
        cout << "Executing Algorithm A1: Single Core Pipeline Baseline...\n";
        run_a1_single_core_pipeline(input_path, output_path);
    } 
    else {
        cerr << "Error: Unknown algorithm identifier '" << algo << "'\n";
        cerr << "Usage: ./video_pipeline [A0|A1|A2...]\n";
        return -1;
    }

    return 0;
}