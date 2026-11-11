#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std;

template <typename T>

class BoundedQueue{
    public:
        BoundedQueue(size_t max_size) : max_size_(max_size), finished_(false) {}

        bool push(const T& item){
            unique_lock<mutex> lock(mtx_);
            
            // wait unitl there is space in the queue or finished is true
            cv_push_.wait(lock, [this](){ return queue_.size() < max_size_ || finished_; });
            
            if (finished_) return false; // if finished, do not push new items

            queue_.push(item);
            cv_pop_.notify_one(); // notify one waiting pop
            return true;
        }

        bool pop(T& item){
            unique_lock<mutex> lock(mtx_);
            
            // wait until there is an item in the queue or finished is true
            cv_pop_.wait(lock, [this](){ return !queue_.empty() || finished_; });
            
            if (queue_.empty()) return false; // if finished and queue is empty, return false

            item = queue_.front();
            queue_.pop();
            cv_push_.notify_one(); // notify one waiting push
            return true;
        }

        void finish(){
            unique_lock<mutex> lock(mtx_);
            finished_ = true;
            cv_push_.notify_all(); // notify all waiting push
            cv_pop_.notify_all();  // notify all waiting pop
        }

    private:
        queue<T> queue_;
        size_t max_size_;
        bool finished_;
        mutex mtx_;
        condition_variable cv_push_;
        condition_variable cv_pop_;


};