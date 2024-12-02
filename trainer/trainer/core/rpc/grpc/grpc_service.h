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

#include "glog/logging.h"

#include "trainer/core/rpc/grpc/grpc_bytebuffer_stream.h"
#include "trainer/core/rpc/tensor_response.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

template <>
class SerializationTraits<sniper::rpc::TensorResponse> {
 public:
  static Status Serialize(const sniper::rpc::TensorResponse& msg,
                          grpc_byte_buffer** bp, bool* own_buffer) {
    return Status();
  }

  static Status Deserialize(grpc_byte_buffer* buffer,
                            sniper::rpc::TensorResponse* msg,
                            int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }

    Status result = g_core_codegen_interface->ok();
    if (result.ok()) {
      sniper::rpc::GrpcByteSource source(buffer);
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
  kUpdatePsShard,

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
      return "/sniper.Sniper/Create";
    case GrpcMethod::kFeedSample:
      return "/sniper.Sniper/FeedSample";
    case GrpcMethod::kPull:
      return "/sniper.Sniper/Pull";
    case GrpcMethod::kPush:
      return "/sniper.Sniper/Push";
    case GrpcMethod::kEmbeddingLookup:
      return "/sniper.Sniper/EmbeddingLookup";
    case GrpcMethod::kPushGrad:
      return "/sniper.Sniper/PushGrad";
    case GrpcMethod::kSave:
      return "/sniper.Sniper/Save";
    case GrpcMethod::kRestore:
      return "/sniper.Sniper/Restore";
    case GrpcMethod::kFreeze:
      return "/sniper.Sniper/Freeze";
    case GrpcMethod::kComplete:
      return "/sniper.Sniper/Complete";
    case GrpcMethod::kCountFeature:
      return "/sniper.Sniper/CountFeature";
    case GrpcMethod::kSaveFeatureCount:
      return "/sniper.Sniper/SaveFeatureCount";
    case GrpcMethod::kRestoreFeatureCount:
      return "/sniper.Sniper/RestoreFeatureCount";
    // hub
    case GrpcMethod::kStartSample:
      return "/sniper.Sniper/StartSample";
    case GrpcMethod::kReadSample:
      return "/sniper.Sniper/ReadSample";
    case GrpcMethod::kUpdateHubShard:
      return "/sniper.Sniper/UpdateHubShard";
    case GrpcMethod::kUpdatePsShard:
      return "/sniper.Sniper/UpdatePsShard";
    // scheduler
    case GrpcMethod::kHeartbeat:
      return "/sniper.Sniper/Heartbeat";
    // Shouldn't be reached.
    case GrpcMethod::kUnknown:
      LOG(FATAL) << "invalid id: not found valid method name";
      abort();
  }

  return nullptr;
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
