#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

namespace sniper {

class Semaphore {
private:
  std::mutex mutex_;
  std::condition_variable condition_;

  uint64_t count_ = 0;

public:
  void Release(uint64_t count = 1) {
    std::lock_guard<decltype(mutex_)> lock(mutex_);
    count_ += count;
    condition_.notify_one();
  }

  void Acquire(uint64_t count = 1) {
    std::unique_lock<decltype(mutex_)> lock(mutex_);
    condition_.wait(lock, [&]{ return count_ >= count; });
    count_ -= count;
  }
};

class SemaphoreLoop {
 public:
  static SemaphoreLoop& GetInstance() {
    static SemaphoreLoop instance;
    return instance;
  }

  bool IsStart();
  void SetIsStart(bool v);
  void Acquire(int idx, uint64_t count = 1);
  void Release(int idx, uint64_t count = 1);
  void ReleaseAll(uint64_t count = 1);

 private:
  SemaphoreLoop();
  std::vector<Semaphore> arr_;
  bool is_start_ = true;
  std::mutex mu_;

};

}  // end namespace sniper
