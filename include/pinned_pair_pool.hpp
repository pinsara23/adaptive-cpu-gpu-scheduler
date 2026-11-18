#pragma once
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "gpu_blur.hpp"

// Stage 6 (A8): K pinned host buffer pairs (input + output, `bytes` each).
// A frame owns exactly one pair from the resize stage (writes `in`) through the GPU
// (DMA reads `in`, DMA writes `out`) to the tail stage (reads `out`), then the pair
// returns here. acquire() blocks while all pairs are in use: this is A8's
// back-pressure point, and the time spent there is reported as "pool wait".
class PinnedPairPool {
public:
    PinnedPairPool(int num_pairs, size_t bytes) {
        if (num_pairs < 1) throw std::invalid_argument("PinnedPairPool: need >= 1 pair");
        for (int i = 0; i < num_pairs; ++i) {
            in_.push_back(std::make_unique<PinnedBuffer>(bytes));
            out_.push_back(std::make_unique<PinnedBuffer>(bytes));
            // Touch every page now (setup), not inside the timed run.
            std::memset(in_.back()->data(), 0, bytes);
            std::memset(out_.back()->data(), 0, bytes);
            free_.push_back(i);
        }
    }

    int size() const { return static_cast<int>(in_.size()); }

    int acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !free_.empty(); });
        const int p = free_.back();
        free_.pop_back();
        return p;
    }

    void release(int p) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            free_.push_back(p);
        }
        cv_.notify_one();
    }

    uint8_t* in(int p)  { return in_[p]->data(); }
    uint8_t* out(int p) { return out_[p]->data(); }

private:
    std::vector<std::unique_ptr<PinnedBuffer>> in_, out_;
    std::vector<int> free_;
    std::mutex mtx_;
    std::condition_variable cv_;
};