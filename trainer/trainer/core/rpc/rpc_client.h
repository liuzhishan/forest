#pragma once

#include <condition_variable>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "trainer/core/base/status.h"
#include "trainer/core/proto/service.pb.h"
#include "trainer/core/rpc/rpc_handler.h"
#include "trainer/core/rpc/tensor_response.h"
#include "tensorflow/core/framework/tensor.h"

DECLARE_int32(rpc_deadline);
DECLARE_int32(rpc_retry_times);

namespace sniper {
namespace rpc {

class RPCClient {
 public:
  RPCClient() {}
  virtual ~RPCClient() {}

  virtual RpcHandlePtr CreateAsync(
      const std::string& ep, const std::string& varname,
      const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr FeedSampleAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr PullAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr PushAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr EmbeddingLookupAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr PushGradAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr SaveAsync(
      const std::string& ep, const std::string& varname,
      const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr RestoreAsync(
      const std::string& ep, const std::string& varname,
      const ::google::protobuf::Message& options, TensorResponse* response,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr FreezeAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr CompleteAsync(
      const std::string& ep, int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr StartSampleAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;
  virtual RpcHandlePtr ReadSampleAsync(
      const std::string& ep, TensorResponse* response,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr UpdateHubShardAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr UpdatePsShardAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr HeartbeatAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr CountFeatureAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr SaveFeatureCountAsync(const std::string& ep,
                                             const std::string& varname,
                                             const ::google::protobuf::Message& options,
                                             TensorResponse* response,
                                             int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  virtual RpcHandlePtr RestoreFeatureCountAsync(const std::string& ep,
                                                const std::string& varname,
                                                const ::google::protobuf::Message& options,
                                                TensorResponse* response,
                                                int64_t timeout_in_ms = FLAGS_rpc_deadline) = 0;

  // Complete tells all the pserver instances that finishe the training,
  // the pserver can reduce it's barrier count, and continue to train
  // with other trainers.
  virtual void SendComplete() = 0;

  template <typename T>
  static RPCClient* GetInstance(int role_id) {
    std::call_once(init_flag_, &RPCClient::Init<T>, role_id);
    return rpc_client_.get();
  }

  // Init is called by GetInstance.
  template <typename T>
  static void Init(int role_id) {
    LOG(INFO) << "init rpc client with role_id: " << role_id;
    role_id_ = role_id;
    if (rpc_client_.get() == nullptr) {
      rpc_client_.reset(new T());
      rpc_client_->InitImpl();
    }
  }

  virtual void InitImpl() {}

 protected:
  // each trainer have exact one trainer id, it should be static
  static int role_id_;

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<RPCClient> rpc_client_;
};

}  // namespace rpc
}  // namespace sniper
