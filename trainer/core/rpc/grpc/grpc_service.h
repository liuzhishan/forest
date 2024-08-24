#pragma once

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>
#include <grpc++/support/byte_buffer.h>

#include "core/rpc/grpc/grpc_bytebuffer_stream.h"
#include "core/rpc/tensor_response.h"
#include "spdlog/spdlog.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

template <>
class SerializationTraits<klearn::rpc::TensorResponse> {
 public:
  static Status Serialize(const klearn::rpc::TensorResponse& msg,
                          grpc_byte_buffer** bp, bool* own_buffer) {
    return Status();
  }

  static Status Deserialize(grpc_byte_buffer* buffer,
                            klearn::rpc::TensorResponse* msg,
                            int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }

    Status result = g_core_codegen_interface->ok();
    if (result.ok()) {
      klearn::rpc::GrpcByteSource source(buffer);
      int ret = msg->Parse(&source);
      if (ret != 0) {
        result = Status(StatusCode::INTERNAL, "TensorResponse parse error");
      }
    }
    g_core_codegen_interface->grpc_byte_buffer_destroy(buffer);
    return result;
  }
};

}  // namespace grpc

namespace sniper {
namespace rpc {

enum class GrpcMethod {
  kCreate,
  kFeedSample,
  kPull,
  kPush,
  kEmbeddingLookup,
  kPushGrad,
  kSave,
  kRestore,
  kFreeze,
  kComplete,

  kStartSample,
  kReadSample,
  kUpdateHubShard,

  kHeartbeat,

  kCountFeature,
  kSaveFeatureCount,
  kRestoreFeatureCount,

  kUnknown
};

static const int kGrpcNumMethods = static_cast<int>(GrpcMethod::kUnknown);

inline const char* GrpcMethodName(GrpcMethod id) {
  switch (id) {
    // ps
    case GrpcMethod::kCreate:
      return "/klearn.KlearnService/Create";
    case GrpcMethod::kFeedSample:
      return "/klearn.KlearnService/FeedSample";
    case GrpcMethod::kPull:
      return "/klearn.KlearnService/Pull";
    case GrpcMethod::kPush:
      return "/klearn.KlearnService/Push";
    case GrpcMethod::kEmbeddingLookup:
      return "/klearn.KlearnService/EmbeddingLookup";
    case GrpcMethod::kPushGrad:
      return "/klearn.KlearnService/PushGrad";
    case GrpcMethod::kSave:
      return "/klearn.KlearnService/Save";
    case GrpcMethod::kRestore:
      return "/klearn.KlearnService/Restore";
    case GrpcMethod::kFreeze:
      return "/klearn.KlearnService/Freeze";
    case GrpcMethod::kComplete:
      return "/klearn.KlearnService/Complete";
    case GrpcMethod::kCountFeature:
      return "/klearn.KlearnService/CountFeature";
    case GrpcMethod::kSaveFeatureCount:
      return "/klearn.KlearnService/SaveFeatureCount";
    case GrpcMethod::kRestoreFeatureCount:
      return "/klearn.KlearnService/RestoreFeatureCount";
    // hub
    case GrpcMethod::kStartSample:
      return "/klearn.KlearnService/StartSample";
    case GrpcMethod::kReadSample:
      return "/klearn.KlearnService/ReadSample";
    case GrpcMethod::kUpdateHubShard:
      return "/klearn.KlearnService/UpdateHubShard";
    // scheduler
    case GrpcMethod::kHeartbeat:
      return "/klearn.KlearnService/Heartbeat";
    // Shouldn't be reached.
    case GrpcMethod::kUnknown:
      SPDLOG_CRITICAL("Invalid id: not found valid method name");
      abort();
  }
  return NULL;
}

class GrpcService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService() {
      for (int i = 0; i < kGrpcNumMethods; ++i) {
        AddMethod(new ::grpc::internal::RpcServiceMethod(
            GrpcMethodName(static_cast<GrpcMethod>(i)),
            ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
        ::grpc::Service::MarkMethodAsync(i);
      }
    }
    virtual ~AsyncService() {}

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace rpc
}  // namespace sniper
