#include <math.h>
#include <inttypes.h>
#include <cassert>
#include <limits>
#include <map>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "trainer/core/util/monitor/run_status.h"

DEFINE_int32(metric_print_interval_in_sec, 30, "");

namespace sniper {
namespace monitor {

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

RunStatus::~RunStatus() { Stop(); }

RunStatus::RunStatus() {
  Init();
  Start();
}

RunStatus* RunStatus::Instance() {
  static RunStatus default_run_status;
  return &default_run_status;
}

int RunStatus::Init() {
  LOG(INFO) << "run status init";
  return 0;
}

int RunStatus::Start() {
  stop_.store(false);
  metric_thread_ = std::thread(&RunStatus::run, this);
  LOG(INFO) << "run status start";
  return 0;
}

void RunStatus::Stop() {
  stop_.store(true);
  cond_.notify_all();
  if (metric_thread_.joinable()) {
    metric_thread_.join();
  }

  LOG(INFO) << "run status stop";
}

void RunStatus::PushTime(monitor::HistogramType type, int64_t time) {
  if (time > 0) {
    statistics_.PushTime(type, static_cast<uint64_t>(time));
  }
}

void RunStatus::run() {
  while (!stop_.load()) {
    printStatistics();
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait_for(lock,
                   std::chrono::seconds(FLAGS_metric_print_interval_in_sec));
  }
}

void RunStatus::printStatistics() {
  LOG(INFO) << "[RunStatus]: " << statistics_.ToString();

  statistics_.Reset();
}

void RunPushTime(HistogramType type, duration<double> time) {
  auto diff_count = static_cast<uint64_t>(duration_cast<microseconds>(time).count());
  if (diff_count > 0) {
    RunStatus::Instance()->PushTime(type, diff_count);
  }
}

void RunPushTime(HistogramType type, uint64_t time) {
  if (time > 0) {
    RunStatus::Instance()->PushTime(type, time);
  }
}

}  // namespace monitor
}  // namespace sniper
