#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "grpc++/channel.h"
#include "grpc++/generic/generic_stub.h"
#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "trainer/core/base/config.h"
#include "trainer/core/base/threadpool.h"
#include "trainer/core/proto/service.grpc.pb.h"
#include "trainer/core/rpc/rpc_client.h"
#include "trainer/core/rpc/tensor_response.h"

namespace sniper {
namespace rpc {

class BaseProcessor {
 public:
  BaseProcessor(std::shared_ptr<grpc::Channel> ch, const std::string& method,
                const ::grpc::ByteBuffer& req, size_t max_retries,
                RpcHandlePtr h, int64_t timeout, grpc::CompletionQueue* cq)
      : cq_(cq),
        stub_g_(ch),
        method_(method),
        req_(req),
        max_retries_(max_retries),
        rpc_h_(h),
        timeout_(timeout) {
    context_ = nullptr;
  }

  virtual ~BaseProcessor() {}

  virtual void Prepare() {
    context_.reset(new grpc::ClientContext());
    context_->set_wait_for_ready(true);
    if (timeout_) {
      std::chrono::system_clock::time_point deadline =
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(timeout_);
      context_->set_deadline(deadline);
    }
  }

  virtual void StartCall() {
    auto call = stub_g_.PrepareUnaryCall(context_.get(), method_, req_, cq_);
    call->StartCall();
    call->Finish(&reply_, &status_, reinterpret_cast<void*>(this));
  }

  void OnCompleted(bool ok) {
    if (status_.ok() && !ok) {
      LOG(INFO) << "GRPC status is okay but CompletionQueueStatus is "
                << "not. This should never happen.";
    }

    if (status_.ok()) {
      Process();
      return;
    }

    // Retry if we have any attempts left
    if (++num_retries_ <= max_retries_ &&
        (status_.error_code() == grpc::StatusCode::UNAVAILABLE)) {
      LOG(INFO) << "Retrying call for " << method_
                << ", Retry: " << num_retries_
                << " of " << max_retries_;

      Prepare();
      StartCall();
    } else {
      LOG(ERROR) << method_ << " meets grpc error, error_code: "
                 << status_.error_code()
                 << ", error_message: " << status_.error_message()
                 << "error_details: " << status_.error_details();

      Finish(false, status_.error_message());
      delete this;
    }
  }

  void Process() {
    ProcessImpl();
    Finish(true);
    delete this;
  }

  RpcHandlePtr GetRpcHandlePtr() { return rpc_h_; }
  bool Wait() { return rpc_h_->Wait(); }
  void Finish(bool ok, const std::string& errmsg = "ok") {
    return rpc_h_->Finish(ok, errmsg);
  }
  virtual void ProcessImpl() = 0;

  grpc::CompletionQueue* cq_;
  std::unique_ptr<grpc::ClientContext> context_;

  ::grpc::GenericStub stub_g_;
  std::string method_;
  ::grpc::ByteBuffer req_;
  ::grpc::ByteBuffer reply_;
  ::grpc::Status status_;

  size_t num_retries_ = 0;
  size_t max_retries_;

 protected:
  RpcHandlePtr rpc_h_;
  int64_t timeout_;
};

class VoidProcessor : public BaseProcessor {
 public:
  explicit VoidProcessor(std::shared_ptr<grpc::Channel> ch,
                         const std::string& method,
                         const ::grpc::ByteBuffer& req, size_t max_retries,
                         RpcHandlePtr h, int64_t timeout,
                         grpc::CompletionQueue* cq)
      : BaseProcessor(ch, method, req, max_retries, h, timeout, cq) {}
  virtual ~VoidProcessor() {}

  void ProcessImpl() override {}
};

void ProcMsgResponse(const grpc::ByteBuffer& buf, TensorResponse* response);

typedef std::function<void(const grpc::ByteBuffer& buf,
                           TensorResponse* response)>
    RequestMsgCallBack;
class MsgProcessor : public BaseProcessor {
 public:
  explicit MsgProcessor(std::shared_ptr<grpc::Channel> ch,
                        const std::string& method,
                        const ::grpc::ByteBuffer& req, size_t max_retries,
                        RpcHandlePtr h, int64_t timeout,
                        grpc::CompletionQueue* cq, TensorResponse* response)
      : BaseProcessor(ch, method, req, max_retries, h, timeout, cq),
        response_(response) {}

  virtual ~MsgProcessor() {}

  void ProcessImpl() override {
    if (response_call_back_) {
      response_call_back_(reply_, response_);
    }
  }

  TensorResponse* response_;
  RequestMsgCallBack response_call_back_ = ProcMsgResponse;
};

class GRPCClient : public RPCClient {
 public:
  GRPCClient()
      : cqs_(15),
        client_threads_(15),
        completed_(false),
        pool_(30),
        stopped_(false) {}
  virtual ~GRPCClient();

  RpcHandlePtr CreateAsync(const std::string& ep, const std::string& varname,
                           const ::google::protobuf::Message& options,
                           int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr FeedSampleAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr PullAsync(const std::string& ep, uint64_t batch_id,
                         const std::string& varname,
                         const ::google::protobuf::Message& options,
                         const tensorflow::Tensor& val1,
                         const tensorflow::Tensor& val2,
                         TensorResponse* response,
                         int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr PushAsync(const std::string& ep, uint64_t batch_id,
                         const std::string& varname,
                         const ::google::protobuf::Message& options,
                         const tensorflow::Tensor& val1,
                         const tensorflow::Tensor& val2,
                         int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr EmbeddingLookupAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      TensorResponse* response,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr PushGradAsync(
      const std::string& ep, uint64_t batch_id, const std::string& varname,
      const ::google::protobuf::Message& options,
      const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr SaveAsync(const std::string& ep, const std::string& varname,
                         const ::google::protobuf::Message& options,
                         int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr RestoreAsync(
      const std::string& ep, const std::string& varname,
      const ::google::protobuf::Message& options, TensorResponse* response,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr FreezeAsync(const std::string& ep,
                           const ::google::protobuf::Message& options,
                           int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr CompleteAsync(
      const std::string& ep,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr StartSampleAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;
  RpcHandlePtr ReadSampleAsync(
      const std::string& ep, TensorResponse* response,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr UpdateHubShardAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr UpdatePsShardAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr HeartbeatAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr CountFeatureAsync(
      const std::string& ep, const ::google::protobuf::Message& options,
      TensorResponse* response, int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr SaveFeatureCountAsync(const std::string& ep,
                                     const std::string& varname,
                                     const ::google::protobuf::Message& options,
                                     TensorResponse* response,
                                     int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  RpcHandlePtr RestoreFeatureCountAsync(const std::string& ep,
                                        const std::string& varname,
                                        const ::google::protobuf::Message& options,
                                        TensorResponse* response,
                                        int64_t timeout_in_ms = FLAGS_rpc_deadline) override;

  void SendComplete() override;

  void InitImpl() override;

 private:
  void Proceed(grpc::CompletionQueue* cq);

  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);
  grpc::CompletionQueue* GetCq() {
    int64_t i = idx.fetch_add(1);
    return &(cqs_[i % cqs_.size()]);
  }

 private:
  std::atomic<int64_t> idx{0};
  std::vector<grpc::CompletionQueue> cqs_;
  std::vector<std::unique_ptr<std::thread>> client_threads_;

  std::unordered_map<std::string, std::shared_ptr<grpc::Channel>> channels_;

  // mutex for GetChannel thread safety
  std::mutex chan_mutex_;
  SNIPER_NOT_COPYABLE_AND_MOVABLE(GRPCClient)

  // mutex for sending complete message only once
  std::mutex completed_mutex_;
  bool completed_;

  ThreadPool pool_;
  volatile bool stopped_;
};

}  // namespace rpc
}  // namespace sniper
