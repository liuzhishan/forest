#pragma once

#include <time.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "trainer/core/base/config.h"

namespace sniper {
namespace rpc {

// client
constexpr char kCreateRPC[] = "CreateRPC";
constexpr char kFeedSampleRPC[] = "FeedSampleRPC";
constexpr char kPullRPC[] = "PullRPC";
constexpr char kPushRPC[] = "PushRPC";
constexpr char kEmbeddingLookupRPC[] = "EmbeddingLookupRPC";
constexpr char kPushGradRPC[] = "PushGradRPC";
constexpr char kSaveRPC[] = "SaveRPC";
constexpr char kRestoreRPC[] = "RestoreRPC";
constexpr char kFreezeRPC[] = "FreezeRPC";
constexpr char kCompleteRPC[] = "CompleteRPC";

constexpr char kStartSampleRPC[] = "StartSampleRPC";
constexpr char kReadSampleRPC[] = "ReadSampleRPC";
constexpr char kUpdateHubShardRPC[] = "UpdateHubShardRPC";

constexpr char kHeartbeatRPC[] = "HeartbeatRPC";
constexpr char kCountFeatureRPC[] = "CountFeatureRPC";
constexpr char kSaveFeatureCountRPC[] = "SaveFeatureCountRPC";
constexpr char kRestoreFeatureCountRPC[] = "RestoreFeatureCountRPC";

class RPCServer;

// client rpc call handle
class RpcHandle {
 public:
  RpcHandle(const std::string ep, const std::string& method,
            const std::string& name)
      : ep_(ep), method_(method), name_(name), status_(kDefaultState) {}

  virtual ~RpcHandle() {}

 public:
  bool should_retry = false;

  bool Wait() {
    int ret = kDefaultState;
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      wait_cond_.wait(lk, [this] { return status_ != kDefaultState; });
      ret = status_;
    }
    return ret != kErrorState;
  }

  void Finish(bool ok, const std::string& errmsg = "ok") {
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      status_ = ok ? kFinishState : kErrorState;
      if (status_ == kErrorState) {
        errmsg_ = errmsg;
      }
    }
    wait_cond_.notify_all();
  }

  std::string String() const {
    std::ostringstream s;
    s << method_ << " name:[" << name_ << "], ep:[" << ep_ << "], status:["
      << status_ << "], errmsg:[" << errmsg_ << "]";
    return s.str();
  }

  std::string ep() const { return ep_; }
  std::string name() const { return name_; }
  std::string method() const { return method_; }
  std::string errmsg() const { return errmsg_; }

 protected:
  // RPC endpoint.
  std::string ep_;
  // RPC method name.
  std::string method_;
  // Variable name.
  std::string name_;
  // RPC error msg.
  std::string errmsg_;

 protected:
  std::mutex sync_mutex_;
  std::condition_variable wait_cond_;

  enum VarHandleStatus {
    kDefaultState = -1,
    kErrorState = 0,
    kFinishState = 1,
  };
  VarHandleStatus status_;

 private:
  SNIPER_NOT_COPYABLE_AND_MOVABLE(RpcHandle)
};

typedef std::shared_ptr<RpcHandle> RpcHandlePtr;

}  // namespace rpc
}  // namespace sniper
