#pragma once

#include <time.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "trainer/core/base/config.h"
#include "trainer/core/proto/meta.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace sniper {
namespace rpc {

// server
constexpr char kRequestCreate[] = "RequestCreate";
constexpr char kRequestFeedSample[] = "RequestFeedSample";
constexpr char kRequestPull[] = "RequestPull";
constexpr char kRequestPush[] = "RequestPush";
constexpr char kRequestEmbeddingLookup[] = "RequestEmbeddingLookup";
constexpr char kRequestPushGrad[] = "RequestPushGrad";
constexpr char kRequestSave[] = "RequestSave";
constexpr char kRequestRestore[] = "RequestRestore";
constexpr char kRequestFreeze[] = "RequestFreeze";
constexpr char kRequestComplete[] = "RequestComplete";
constexpr char kRequestCountFeature[] = "RequestCountFeature";
constexpr char kRequestSaveFeatureCount[] = "RequestSaveFeatureCount";
constexpr char kRequestRestoreFeatureCount[] = "RequestRestoreFeatureCount";

// hub
constexpr char kRequestStartSample[] = "RequestStartSample";
constexpr char kRequestReadSample[] = "RequestReadSample";
constexpr char kRequestUpdateHubShard[] = "RequestUpdateHubShard";

// scheduler
constexpr char kRequestHeartbeat[] = "RequestHeartbeat";

enum DistributedMode { kSync = 0, kAsync = 1, kHalfAsync = 2, kGeo = 3 };

class RPCServer;

class RequestHandler {
 public:
  explicit RequestHandler(int distributed_mode)
      : distributed_mode_(distributed_mode), rpc_server_(nullptr) {}

  virtual ~RequestHandler() {}

  void SetRPCServer(RPCServer* rpc_server) { rpc_server_ = rpc_server; }

  // Get attributes.
  int distributed_mode() { return distributed_mode_; }

  virtual bool Handle(
      Role role, int32_t role_id, uint64_t seq_id, const std::string& var_name,
      const ::google::protobuf::Any& options, const tensorflow::Tensor& val1,
      const tensorflow::Tensor& val2, ::google::protobuf::Message* out_options,
      tensorflow::Tensor* out_val1, tensorflow::Tensor* out_val2) = 0;

 protected:
  const int distributed_mode_;
  RPCServer* rpc_server_;
};

}  // namespace rpc
}  // namespace sniper
