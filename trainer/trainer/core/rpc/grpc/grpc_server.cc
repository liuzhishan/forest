#include "trainer/core/rpc/grpc/grpc_server.h"

#include <unistd.h>

#include <chrono>
#include <limits>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "trainer/core/rpc/grpc/grpc_tensor_coding.h"

using ::grpc::ServerAsyncResponseWriter;

DEFINE_bool(rpc_server_disable_reuse_port, false, "");
DEFINE_string(grpc_server_compression_algo, "", "");
DEFINE_int32(grpc_server_compression_level, 3, "");

namespace sniper {
namespace rpc {
enum CallStatus { PROCESS = 0, FINISH };

class RequestBase {
 public:
  explicit RequestBase(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    CHECK(cq_ != nullptr);
  }
  virtual ~RequestBase() {}
  virtual void Process() = 0;

  std::string Status2String(const std::string& method) {
    std::string status = "Process";
    if (status_ == FINISH) {
      status = "Finish";
    }

    std::ostringstream s;
    s << method << ", ep:[" << ctx_.peer() << "]"
      << " " << status << " using req_id:" << req_id_;
    return s.str();
  }

  CallStatus Status() const {
    std::lock_guard<std::mutex> l(status_mu_);
    return status_;
  }

  template <typename T>
  void Finish(const T& reply, ServerAsyncResponseWriter<T>* responder) {
    std::lock_guard<std::mutex> l(status_mu_);
    status_ = FINISH;
    responder->Finish(reply, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  mutable std::mutex status_mu_;
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  RequestHandler* request_handler_;
  int req_id_;
};

class RequestCreate final : public RequestBase {
 public:
  explicit RequestCreate(GrpcService::AsyncService* service,
                         ::grpc::ServerCompletionQueue* cq,
                         RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kCreate);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestCreate() {}

  void Process() override {
    auto& meta = request_->meta();
    CreateOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestFeedSample final : public RequestBase {
 public:
  explicit RequestFeedSample(GrpcService::AsyncService* service,
                             ::grpc::ServerCompletionQueue* cq,
                             RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kFeedSample);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestFeedSample() {}

  void Process() override {
    auto& meta = request_->meta();
    std::string s;
    meta.SerializeToString(&s);
    FeedSampleOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestPush final : public RequestBase {
 public:
  explicit RequestPush(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kPush);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestPush() {}

  void Process() override {
    auto& meta = request_->meta();
    PushOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestPull final : public RequestBase {
 public:
  explicit RequestPull(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    auto method_id = static_cast<int>(GrpcMethod::kPull);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestPull() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    PullOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, &out_val1, &out_val2);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestEmbeddingLookup final : public RequestBase {
 public:
  explicit RequestEmbeddingLookup(GrpcService::AsyncService* service,
                                  ::grpc::ServerCompletionQueue* cq,
                                  RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    auto method_id = static_cast<int>(GrpcMethod::kEmbeddingLookup);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestEmbeddingLookup() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    EmbeddingLookupOption out_options;
    auto start = std::chrono::high_resolution_clock::now();
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, &out_val1, &out_val2);
    auto end = std::chrono::high_resolution_clock::now();
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestPushGrad final : public RequestBase {
 public:
  explicit RequestPushGrad(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kPushGrad);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestPushGrad() {}

  void Process() override {
    auto& meta = request_->meta();
    PushGradOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestSave : public RequestBase {
 public:
  explicit RequestSave(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kSave);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestSave() {}

  void Process() override {
    auto& meta = request_->meta();
    SaveOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestRestore final : public RequestBase {
 public:
  explicit RequestRestore(GrpcService::AsyncService* service,
                          ::grpc::ServerCompletionQueue* cq,
                          RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kRestore);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestRestore() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    RestoreOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, &out_val1, &out_val2);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestFreeze final : public RequestBase {
 public:
  explicit RequestFreeze(GrpcService::AsyncService* service,
                         ::grpc::ServerCompletionQueue* cq,
                         RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kFreeze);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestFreeze() {}

  void Process() override {
    auto& meta = request_->meta();
    FreezeOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestComplete final : public RequestBase {
 public:
  explicit RequestComplete(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kComplete);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestComplete() {}

  void Process() override {
    auto& meta = request_->meta();
    CompleteOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestStartSample final : public RequestBase {
 public:
  explicit RequestStartSample(GrpcService::AsyncService* service,
                              ::grpc::ServerCompletionQueue* cq,
                              RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    auto method_id = static_cast<int>(GrpcMethod::kStartSample);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestStartSample() {}

  void Process() override {
    auto& meta = request_->meta();
    StartSampleOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestReadSample final : public RequestBase {
 public:
  explicit RequestReadSample(GrpcService::AsyncService* service,
                             ::grpc::ServerCompletionQueue* cq,
                             RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    auto method_id = static_cast<int>(GrpcMethod::kReadSample);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestReadSample() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    ReadSampleOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, &out_val1, &out_val2);
    EncodeTensorToByteBuffer(ROLE_HUB, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestUpdateHubShard final : public RequestBase {
 public:
  explicit RequestUpdateHubShard(GrpcService::AsyncService* service,
                                 ::grpc::ServerCompletionQueue* cq,
                                 RequestHandler* request_handler,
                                 int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    auto method_id = static_cast<int>(GrpcMethod::kUpdateHubShard);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestUpdateHubShard() {}

  void Process() override {
    auto& meta = request_->meta();
    UpdateHubShardOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    Finish(reply_, &responder_);
  }

 protected:
  VoidMessage reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<VoidMessage> responder_;
};

class RequestHeartbeat final : public RequestBase {
 public:
  explicit RequestHeartbeat(GrpcService::AsyncService* service,
                            ::grpc::ServerCompletionQueue* cq,
                            RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kHeartbeat);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestHeartbeat() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    HeartbeatOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestCountFeature final : public RequestBase {
 public:
  explicit RequestCountFeature(GrpcService::AsyncService* service,
                               ::grpc::ServerCompletionQueue* cq,
                               RequestHandler* request_handler,
                               int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kCountFeature);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestCountFeature() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    CountFeatureOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestSaveFeatureCount final : public RequestBase {
 public:
  explicit RequestSaveFeatureCount(GrpcService::AsyncService* service,
                                   ::grpc::ServerCompletionQueue* cq,
                                   RequestHandler* request_handler,
                                   int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kSaveFeatureCount);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestSaveFeatureCount() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    SaveFeatureCountOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestRestoreFeatureCount final : public RequestBase {
 public:
  explicit RequestRestoreFeatureCount(GrpcService::AsyncService* service,
                                      ::grpc::ServerCompletionQueue* cq,
                                      RequestHandler* request_handler,
                                      int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new TensorResponse());
    int method_id = static_cast<int>(GrpcMethod::kRestoreFeatureCount);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestRestoreFeatureCount() {}

  void Process() override {
    auto& meta = request_->meta();
    tensorflow::Tensor out_val1;
    tensorflow::Tensor out_val2;
    RestoreFeatureCountOption out_options;
    request_handler_->Handle(meta.role(), meta.role_id(), meta.seq_id(),
                             meta.varname(), meta.options(),
                             request_->tensor1(), request_->tensor2(),
                             &out_options, nullptr, nullptr);
    EncodeTensorToByteBuffer(ROLE_PS, 0, meta.seq_id(), meta.varname(),
                             out_options, out_val1, out_val2, &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  ::grpc::ByteBuffer reply_;
  std::shared_ptr<TensorResponse> request_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

void AsyncGRPCServer::WaitServerReady() {
  LOG(INFO) << "AsyncGRPCServer is waiting server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  LOG(INFO) << "AsyncGRPCServer WaitSeverReady";
}

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
// Come from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

void AsyncGRPCServer::StartServer() {
  ::grpc::ServerBuilder builder;
  if (FLAGS_grpc_server_compression_algo == "gzip") {
    builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    builder.SetDefaultCompressionLevel(GRPC_COMPRESS_LEVEL_LOW);
  }
  builder.AddListeningPort(bind_address_, grpc::InsecureServerCredentials());
  builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  if (FLAGS_rpc_server_disable_reuse_port) {
    builder.SetOption(
        std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
    LOG(INFO) << "set FLAGS_rpc_server_disable_reuse_port";
  }
  std::unique_ptr<GrpcService::AsyncService> service(
      new GrpcService::AsyncService());
  builder.RegisterService(service.get());

  for (auto t : rpc_call_map_) {
    rpc_cq_[t.first].reset(builder.AddCompletionQueue().release());
  }

  server_ = builder.BuildAndStart();
  if (server_ != nullptr) {
    LOG(INFO) << "Server listening on " <<  bind_address_ << " successful";
    service_.reset(service.release());
  } else {
    LOG(FATAL) << "Server listening on " << bind_address_ << " successful";
  }

  std::function<void(const std::string&, int)> f =
      std::bind(&AsyncGRPCServer::TryToRegisterNewOne, this,
                std::placeholders::_1, std::placeholders::_2);

  for (auto& t : rpc_call_map_) {
    auto& rpc_name = t.first;
    auto& cq = rpc_cq_[rpc_name];
    auto threadnum = rpc_thread_num_[rpc_name];
    auto& reqs = rpc_reqs_[rpc_name];

    reqs.resize(kRequestBufSize);

    for (int i = 0; i < kRequestBufSize; i++) {
      TryToRegisterNewOne(rpc_name, i);
    }

    for (int i = 0; i < threadnum; i++) {
      rpc_threads_[rpc_name].emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq.get(), rpc_name, f)));
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  // wait server
  server_->Wait();

  for (auto& t : rpc_threads_) {
    auto& threads = t.second;
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i]->join();
      LOG(INFO) << t.first << " threads ends!";
    }
  }
}

void AsyncGRPCServer::ShutdownQueue() {
  for (auto& t : rpc_cq_) {
    t.second->Shutdown();
    LOG(INFO) << t.first << " queue shutdown!";
  }
}

void AsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_shut_down_ = true;
  ShutdownQueue();

  LOG(INFO) << "server shutdown!";
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewOne(const std::string& rpc_name,
                                          int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    LOG(INFO) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }

  auto& reqs = rpc_reqs_[rpc_name];
  auto& handler = rpc_call_map_[rpc_name];
  auto& cq = rpc_cq_[rpc_name];

  RequestBase* b = nullptr;
  if (rpc_name == kRequestCreate) {
    b = new RequestCreate(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestFeedSample) {
    b = new RequestFeedSample(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestPull) {
    b = new RequestPull(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestPush) {
    b = new RequestPush(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestEmbeddingLookup) {
    b = new RequestEmbeddingLookup(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestPushGrad) {
    b = new RequestPushGrad(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestSave) {
    b = new RequestSave(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestRestore) {
    b = new RequestRestore(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestFreeze) {
    b = new RequestFreeze(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestComplete) {
    b = new RequestComplete(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestStartSample) {
    b = new RequestStartSample(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestReadSample) {
    b = new RequestReadSample(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestUpdateHubShard) {
    b = new RequestUpdateHubShard(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestHeartbeat) {
    b = new RequestHeartbeat(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestCountFeature) {
    b = new RequestCountFeature(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestSaveFeatureCount) {
    b = new RequestSaveFeatureCount(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestRestoreFeatureCount) {
    b = new RequestRestoreFeatureCount(service_.get(), cq.get(), handler, req_id);
  } else {
    CHECK(false) << "not supported rpc";
  }
  reqs[req_id] = b;
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
    std::function<void(const std::string&, int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << "CompletionQueue " << rpc_name << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));

    auto& reqs = rpc_reqs_[rpc_name];
    RequestBase* base = nullptr;
    {
      CHECK(req_id >= 0 && req_id < kRequestBufSize);
      std::unique_lock<std::mutex> lock(cq_mutex_);
      base = reqs[req_id];
    }

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      LOG(ERROR) << "completion queue: " << rpc_name
                 << " recv no regular event context: " << base->Status2String(rpc_name);

      TryToRegisterNewOne(rpc_name, req_id);
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        base->Process();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(rpc_name, req_id);
        delete base;
        break;
      }
      default: {
        CHECK(false) << "invalid base status";
      }
    }
  }
}

}  // namespace rpc
}  // namespace sniper
