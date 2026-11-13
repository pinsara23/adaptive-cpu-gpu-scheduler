#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template <typename T>
class BoundedQueue {
public:
    BoundedQueue(size_t max_size) : max_size_(max_size), finished_(false) {
        if (max_size_ == 0) {
            throw std::invalid_argument("Queue capacity must be > 0");
        }
    }

    // Move semantics overload for pushing
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_push_.wait(lock, [this]() { return queue_.size() < max_size_ || finished_; });
        if (finished_) return false;
        
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    // Copy semantics for pushing (fallback)
    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_push_.wait(lock, [this]() { return queue_.size() < max_size_ || finished_; });
        if (finished_) return false;
        
        queue_.push(item);
        cv_pop_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_pop_.wait(lock, [this]() { return !queue_.empty() || finished_; });
        if (queue_.empty() && finished_) return false;
        
        item = std::move(queue_.front()); // Move semantics for pop
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    void finish() {
        std::unique_lock<std::mutex> lock(mtx_);
        finished_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t max_size_;
    bool finished_;
    std::mutex mtx_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
};