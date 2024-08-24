#include <vector>
#include <mutex>
#include <condition_variable>

#include "core/base/semaphore.h"
#include "core/base/util.h"

namespace sniper {

SemaphoreLoop::SemaphoreLoop(): arr_(5) {
  is_start_ = true;
}

bool SemaphoreLoop::IsStart() {
  return is_start_;
}

void SemaphoreLoop::SetIsStart(bool v) {
  std::lock_guard<std::mutex> lk(mu_);
  is_start_ = v;
}

void SemaphoreLoop::Acquire(int idx, uint64_t count) {
  arr_[idx].Acquire(count);
}

void SemaphoreLoop::Release(int idx, uint64_t count) {
  arr_[idx].Release(count);
}

void SemaphoreLoop::ReleaseAll(uint64_t count) {
  for (size_t i = 0; i < arr_.size(); i++) {
    arr_[i].Release(count);
  }
}

}  // end namespace sniper
