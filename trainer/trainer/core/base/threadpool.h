#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "trainer/core/base/config.h"
#include "trainer/core/base/status.h"

namespace sniper {

struct EnforceNotMet : public std::exception {
  EnforceNotMet(std::exception_ptr e, const char* file, int line) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception& e) {
      err_str_ = GetTraceBackString(e.what(), file, line);
    }
  }

  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str.c_str(), file, line)) {}

  EnforceNotMet(const Status& error, const char* file, int line)
      : err_str_(GetTraceBackString(error.ToString().c_str(), file, line)) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

  std::string GetTraceBackString(const char* msg, const char* file, int line) {
    std::ostringstream sout;
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
    sout << msg << " at "
         << "(" << file << ":" << line << ")" << std::endl;
    return sout.str();
  }

  std::string err_str_;
};

struct ExceptionHandler {
  mutable std::future<std::unique_ptr<EnforceNotMet>> future_;
  explicit ExceptionHandler(std::future<std::unique_ptr<EnforceNotMet>>&& f)
      : future_(std::move(f)) {}
  void operator()() const {
    auto ex = this->future_.get();
    if (ex != nullptr) {
      LOG(FATAL) << "The exception is thrown inside the thread pool. You "
                 << "should use RunAndGetException to handle the exception.";
    }
  }
};

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ~ThreadPool();

  using Task = std::packaged_task<std::unique_ptr<EnforceNotMet>()>;

  template <typename Callback>
  std::future<void> Run(Callback fn) {
    auto f = this->RunAndGetException(fn);
    return std::async(std::launch::deferred, ExceptionHandler(std::move(f)));
  }

  template <typename Callback>
  std::future<std::unique_ptr<EnforceNotMet>> RunAndGetException(Callback fn) {
    Task task([fn]() -> std::unique_ptr<EnforceNotMet> {
      try {
        fn();
      } catch (EnforceNotMet& ex) {
        return std::unique_ptr<EnforceNotMet>(new EnforceNotMet(ex));
      } catch (const std::exception& e) {
        LOG(FATAL) << "Unexpected exception is catched in thread pool. All "
                   << "throwable exception should be an EnforceNotMet. " << e.what();
      }
      return nullptr;
    });
    std::future<std::unique_ptr<EnforceNotMet>> f = task.get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!running_) {
        throw EnforceNotMet("enqueue on stopped ThreadPool", __FILE__,
                            __LINE__);
      }
      tasks_.push(std::move(task));
    }
    scheduled_.notify_one();
    return f;
  }

 private:
  SNIPER_NOT_COPYABLE_AND_MOVABLE(ThreadPool);

  void TaskLoop();

  static void Init();

 private:
  std::vector<std::unique_ptr<std::thread>> threads_;

  std::queue<Task> tasks_;
  std::mutex mutex_;
  bool running_;
  std::condition_variable scheduled_;
};

}  // namespace sniper
