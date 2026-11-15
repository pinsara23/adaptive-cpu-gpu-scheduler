#pragma once

namespace config {
    // Working resolution for the CPU pipeline. One definition for the whole project.
    constexpr int TARGET_WIDTH  = 1280;
    constexpr int TARGET_HEIGHT = 720;

    // Compute-only benchmarking mode (--no-encode): frames are fully processed
    // but never written. Default false, so normal runs are unchanged.
    inline bool no_encode = false;
}