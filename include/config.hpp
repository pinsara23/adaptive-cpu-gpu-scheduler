#pragma once

namespace config {
    // Working resolution for the CPU pipeline. One definition for the whole project.
    constexpr int TARGET_WIDTH  = 1280;
    constexpr int TARGET_HEIGHT = 720;

    // A8 (Stage 6): GPU streams of the GPU stage thread (Finding 18: S=2 suffices for
    // blur-only work) and spare pinned pairs beyond R + S + T (override: HPC_A8_SLACK).
    constexpr int A8_GPU_STREAMS = 2;
    constexpr int A8_POOL_SLACK  = 4;

    // Compute-only benchmarking mode (--no-encode): frames are fully processed
    // but never written. Default false, so normal runs are unchanged.
    inline bool no_encode = false;
}