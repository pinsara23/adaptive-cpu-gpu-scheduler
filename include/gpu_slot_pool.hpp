#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "gpu_blur.hpp"

// A fixed pool of S GPU "slots". Each slot = one CUDA stream + its own device buffers
// (a GpuBlur5 without host buffers). A thread acquires a free slot, runs one frame on
// it and releases it, so at most S frames are on the GPU at once and no two frames
// ever share device buffers. acquire() blocks while all slots are busy (a counting
// semaphore built from a mutex, a condition variable and a free list).
class GpuSlotPool {
public:
    GpuSlotPool(int num_slots, int width, int height, int channels) {
        if (num_slots < 1) throw std::invalid_argument("GpuSlotPool: need >= 1 slot");
        for (int i = 0; i < num_slots; ++i) {
            slots_.push_back(std::make_unique<GpuBlur5>(width, height, channels, false));
            free_.push_back(i);
        }
    }

    int size() const { return static_cast<int>(slots_.size()); }

    int acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !free_.empty(); });
        const int s = free_.back();
        free_.pop_back();
        return s;
    }

    void release(int s) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            free_.push_back(s);
        }
        cv_.notify_one();
    }

    GpuBlur5& slot(int s) { return *slots_[s]; }

private:
    std::vector<std::unique_ptr<GpuBlur5>> slots_;
    std::vector<int> free_;
    std::mutex mtx_;
    std::condition_variable cv_;
};