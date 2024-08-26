#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>

#include "trainer/core/util/monitor/statistics.h"

namespace sniper {
namespace monitor {

using std::chrono::duration;

class RunStatus {
 public:
  ~RunStatus();
  static RunStatus* Instance();

  void PushTime(HistogramType type, int64_t time);

 private:
  void run();
  void printStatistics();

 private:
  RunStatus();
  RunStatus(const RunStatus&) = delete;
  RunStatus& operator=(const RunStatus&) = delete;

  int Init();
  int Start();
  void Stop();

  Statistics statistics_;
  std::atomic<bool> stop_ = {true};

  std::mutex mutex_;
  std::condition_variable cond_;
  std::thread metric_thread_;
};

void RunPushTime(HistogramType type, duration<double> time);
void RunPushTime(HistogramType type, uint64_t time);

}  // namespace monitor
}  // namespace sniper
