#include "trainer/core/rpc/grpc/grpc_client.h"

#include <stdlib.h>

#include <limits>

#include "glog/logging.h"
#include "trainer/core/rpc/grpc/grpc_tensor_coding.h"

DEFINE_bool(rpc_client_disable_reuse_port, true, "");
DEFINE_string(grpc_client_compression_algo, "", "");
DEFINE_int32(grpc_client_compression_level, 3, "");

namespace sniper {
namespace rpc {

void ProcMsgResponse(const grpc::ByteBuffer& buf, TensorResponse* response) {
  DecodeTensorFromByteBuffer(response, buf);
}

void GRPCClient::InitImpl() {
  for (size_t i = 0; i < client_threads_.size(); ++i) {
    client_threads_[i].reset(
        new std::thread(std::bind(&GRPCClient::Proceed, this, &(cqs_[i]))));
  }
}

void GRPCClient::SendComplete() {
  std::unique_lock<std::mutex> lk(completed_mutex_);
  if (!completed_) {
    for (auto& it : channels_) {
      LOG(INFO) << "send complete message to " << it.first;
      this->CompleteAsync(it.first, role_id_);
    }
    completed_ = true;
  }
}

GRPCClient::~GRPCClient() {
  stopped_ = true;
  for (auto& cq : cqs_) {
    cq.Shutdown();
  }
  for (auto& t : client_threads_) {
    t->join();
  }
  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    for (auto& it : channels_) {
      it.second.reset();
    }
    channels_.clear();
  }
}

RpcHandlePtr GRPCClient::CreateAsync(const std::string& ep,
                                     const std::string& varname,
                                     const ::google::protobuf::Message& options,
                                     int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname(varname);
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kCreateRPC, "Create"));
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/Create", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::FeedSampleAsync(
    const std::string& ep, uint64_t batch_id, const std::string& varname,
    const ::google::protobuf::Message& options, const tensorflow::Tensor& val1,
    const tensorflow::Tensor& val2, int64_t timeout_in_ms) {
  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kFeedSampleRPC, "FeedSample"));
  pool_.Run(
      [ch, h, batch_id, varname, &options, val1, val2, timeout_in_ms, this] {
        ::grpc::ByteBuffer req;
        std::string s;
        options.SerializeToString(&s);
        EncodeTensorToByteBuffer(ROLE_HUB, role_id_, batch_id, varname, options,
                                 val1, val2, &req);
        VoidProcessor* c =
            new VoidProcessor(ch, "/sniper.SniperService/FeedSample", req,
                              FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
        c->Prepare();
        c->StartCall();
      });
  return h;
}

RpcHandlePtr GRPCClient::PullAsync(const std::string& ep, uint64_t batch_id,
                                   const std::string& varname,
                                   const ::google::protobuf::Message& options,
                                   const tensorflow::Tensor& val1,
                                   const tensorflow::Tensor& val2,
                                   TensorResponse* response,
                                   int64_t timeout_in_ms) {
  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kPullRPC, "Pull"));
  ::grpc::ByteBuffer req;
  EncodeTensorToByteBuffer(ROLE_TRAINER, role_id_, batch_id, varname, options,
                           val1, val2, &req);
  pool_.Run([ch, response, h, req, timeout_in_ms, this] {
    MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/Pull", req,
                                       FLAGS_rpc_retry_times, h, timeout_in_ms,
                                       GetCq(), response);
    c->response_call_back_ = ProcMsgResponse;
    c->Prepare();
    c->StartCall();
  });
  return h;
}

RpcHandlePtr GRPCClient::PushAsync(const std::string& ep, uint64_t batch_id,
                                   const std::string& varname,
                                   const ::google::protobuf::Message& options,
                                   const tensorflow::Tensor& val1,
                                   const tensorflow::Tensor& val2,
                                   int64_t timeout_in_ms) {
  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kPushRPC, "Push"));
  ::grpc::ByteBuffer req;
  EncodeTensorToByteBuffer(ROLE_TRAINER, role_id_, batch_id, varname, options,
                           val1, val2, &req);
  pool_.Run([ch, h, req, timeout_in_ms, this] {
    VoidProcessor* c =
        new VoidProcessor(ch, "/sniper.SniperService/Push", req,
                          FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
    c->Prepare();
    c->StartCall();
  });
  return h;
}

RpcHandlePtr GRPCClient::EmbeddingLookupAsync(
    const std::string& ep, uint64_t batch_id, const std::string& varname,
    const ::google::protobuf::Message& options, const tensorflow::Tensor& val1,
    const tensorflow::Tensor& val2, TensorResponse* response,
    int64_t timeout_in_ms) {
  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kEmbeddingLookupRPC, "EmbeddingLookup"));
  ::grpc::ByteBuffer req;
  EncodeTensorToByteBuffer(ROLE_TRAINER, role_id_, batch_id, varname, options,
                           val1, val2, &req);
  pool_.Run([ch, response, h, req, timeout_in_ms, this] {
    MsgProcessor* c = new MsgProcessor(
        ch, "/sniper.SniperService/EmbeddingLookup", req, FLAGS_rpc_retry_times,
        h, timeout_in_ms, GetCq(), response);
    c->response_call_back_ = ProcMsgResponse;
    c->Prepare();
    c->StartCall();
  });
  return h;
}

RpcHandlePtr GRPCClient::PushGradAsync(
    const std::string& ep, uint64_t batch_id, const std::string& varname,
    const ::google::protobuf::Message& options, const tensorflow::Tensor& val1,
    const tensorflow::Tensor& val2, int64_t timeout_in_ms) {
  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kPushGradRPC, "PushGrad"));
  pool_.Run(
      [ch, h, batch_id, varname, &options, val1, val2, timeout_in_ms, this] {
        ::grpc::ByteBuffer req;
        EncodeTensorToByteBuffer(ROLE_TRAINER, role_id_, batch_id, varname,
                                 options, val1, val2, &req);
        VoidProcessor* c =
            new VoidProcessor(ch, "/sniper.SniperService/PushGrad", req,
                              FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
        c->Prepare();
        c->StartCall();
      });
  return h;
}

RpcHandlePtr GRPCClient::SaveAsync(const std::string& ep,
                                   const std::string& varname,
                                   const ::google::protobuf::Message& options,
                                   int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname(varname);
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kSaveRPC, "Save"));
  const auto ch = GetChannel(ep);
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/Save", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::RestoreAsync(
    const std::string& ep, const std::string& varname,
    const ::google::protobuf::Message& options, TensorResponse* response,
    int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname(varname);
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kRestoreRPC, "Restore"));
  const auto ch = GetChannel(ep);

  MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/Restore", req,
                                     FLAGS_rpc_retry_times, h, timeout_in_ms,
                                     GetCq(), response);
  c->response_call_back_ = ProcMsgResponse;
  c->Prepare();
  c->StartCall();

  return h;
}

RpcHandlePtr GRPCClient::FreezeAsync(const std::string& ep,
                                     const ::google::protobuf::Message& options,
                                     int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kFreezeRPC, "Freeze"));
  const auto ch = GetChannel(ep);
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/Freeze", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::CompleteAsync(const std::string& ep,
                                       int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kCompleteRPC, "Complete"));
  const auto ch = GetChannel(ep);
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/Complete", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::StartSampleAsync(
    const std::string& ep, const ::google::protobuf::Message& options,
    int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kStartSampleRPC, "StartSample"));
  const auto ch = GetChannel(ep);
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/StartSample", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::ReadSampleAsync(const std::string& ep,
                                         TensorResponse* response,
                                         int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  const auto ch = GetChannel(ep);
  RpcHandlePtr h(new RpcHandle(ep, kReadSampleRPC, "ReadSample"));

  pool_.Run([ch, response, h, req, timeout_in_ms, this] {
    MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/ReadSample",
                                       req, FLAGS_rpc_retry_times, h,
                                       timeout_in_ms, GetCq(), response);
    c->response_call_back_ = ProcMsgResponse;
    c->Prepare();
    c->StartCall();
  });
  return h;
}

RpcHandlePtr GRPCClient::UpdateHubShardAsync(
    const std::string& ep, const ::google::protobuf::Message& options,
    int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kUpdateHubShardRPC, "UpdateHubShard"));
  const auto ch = GetChannel(ep);
  VoidProcessor* c =
      new VoidProcessor(ch, "/sniper.SniperService/UpdateHubShard", req,
                        FLAGS_rpc_retry_times, h, timeout_in_ms, GetCq());
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::HeartbeatAsync(
    const std::string& ep, const ::google::protobuf::Message& options,
    TensorResponse* response, int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_PS);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kHeartbeatRPC, "Heartbeat"));
  const auto ch = GetChannel(ep);
  MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/Heartbeat", req,
                                     FLAGS_rpc_retry_times, h, timeout_in_ms,
                                     GetCq(), response);
  c->response_call_back_ = ProcMsgResponse;
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::CountFeatureAsync(
    const std::string& ep, const ::google::protobuf::Message& options,
    TensorResponse* response, int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname("");
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kCountFeatureRPC, "CountFeature"));
  const auto ch = GetChannel(ep);
  MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/CountFeature", req,
                                     FLAGS_rpc_retry_times, h, timeout_in_ms,
                                     GetCq(), response);
  c->response_call_back_ = ProcMsgResponse;
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::SaveFeatureCountAsync(const std::string& ep,
                                               const std::string& varname,
                                               const ::google::protobuf::Message& options,
                                               TensorResponse* response,
                                               int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname(varname);
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kSaveFeatureCountRPC, "SaveFeatureCount"));
  const auto ch = GetChannel(ep);
  MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/SaveFeatureCount", req,
                                     FLAGS_rpc_retry_times, h, timeout_in_ms,
                                     GetCq(), response);
  c->response_call_back_ = ProcMsgResponse;
  c->Prepare();
  c->StartCall();
  return h;
}

RpcHandlePtr GRPCClient::RestoreFeatureCountAsync(const std::string& ep,
                                                  const std::string& varname,
                                                  const ::google::protobuf::Message& options,
                                                  TensorResponse* response,
                                                  int64_t timeout_in_ms) {
  TensorMessage msg;
  msg.set_role(ROLE_TRAINER);
  msg.set_role_id(role_id_);
  msg.set_seq_id(0);
  msg.set_varname(varname);
  msg.mutable_options()->PackFrom(options);

  ::grpc::ByteBuffer req;
  EncodeTensorMessageToByteBuffer(msg, &req);

  RpcHandlePtr h(new RpcHandle(ep, kRestoreFeatureCountRPC, "RestoreFeatureCount"));
  const auto ch = GetChannel(ep);
  MsgProcessor* c = new MsgProcessor(ch, "/sniper.SniperService/RestoreFeatureCount", req,
                                     FLAGS_rpc_retry_times, h, timeout_in_ms,
                                     GetCq(), response);
  c->response_call_back_ = ProcMsgResponse;
  c->Prepare();
  c->StartCall();
  return h;
}

void GRPCClient::Proceed(grpc::CompletionQueue* cq) {
  void* tag = nullptr;
  bool ok = false;
  while (!stopped_ && cq->Next(&tag, &ok)) {
    BaseProcessor* c = static_cast<BaseProcessor*>(tag);
    GPR_ASSERT(ok);
    c->OnCompleted(ok);
  }
}

std::shared_ptr<grpc::Channel> GRPCClient::GetChannel(const std::string& ep) {
  std::lock_guard<std::mutex> guard(chan_mutex_);
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  // Channel configurations:
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 2000);
  if (FLAGS_rpc_client_disable_reuse_port) {
    args.SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  LOG(INFO) << "Setting GRPC compression : algo: " << FLAGS_grpc_client_compression_algo
            << ", level: " << FLAGS_grpc_client_compression_level;

  if (FLAGS_grpc_client_compression_algo == "gzip") {
    args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                FLAGS_grpc_client_compression_level);
  } else {
    args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
  }

  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());

  auto ch =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);
  channels_[ep] = ch;
  LOG(INFO) << "get channel success, ep: " << ep;
  return ch;
}

}  // namespace rpc
}  // namespace sniper
