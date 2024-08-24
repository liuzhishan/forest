#include "core/base/threadpool.h"

#include <memory>
#include <utility>

namespace sniper {

ThreadPool::ThreadPool(int num_threads) : running_(true) {
  threads_.resize(num_threads);
  for (auto& thread : threads_) {
    thread.reset(new std::thread(std::bind(&ThreadPool::TaskLoop, this)));
  }
}

ThreadPool::~ThreadPool() {
  {
    // notify all threads to stop running
    std::unique_lock<std::mutex> l(mutex_);
    running_ = false;
  }
  scheduled_.notify_all();

  for (auto& t : threads_) {
    t->join();
    t.reset(nullptr);
  }
}

void ThreadPool::TaskLoop() {
  while (true) {
    Task task;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      scheduled_.wait(
          lock, [this] { return !this->tasks_.empty() || !this->running_; });

      if (!running_ && tasks_.empty()) {
        return;
      }

      if (tasks_.empty()) {
        throw EnforceNotMet("his thread has no task to Run", __FILE__,
                            __LINE__);
      }

      // pop a task from the task queue
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    // run the task
    task();
  }
}

}  // namespace sniper
