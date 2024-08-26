#pragma once

#include <time.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "trainer/core/rpc/request_handler.h"

namespace sniper {
namespace rpc {

class RequestLookupHandler final : public RequestHandler {
 public:
  explicit RequestLookupHandler(int distributed_mode)
      : RequestHandler(distributed_mode) {}
  virtual ~RequestLookupHandler() {}
  bool Handle(Role role, int32_t role_id, uint64_t seq_id,
              const std::string& var_name,
              const ::google::protobuf::Any& options,
              const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
              ::google::protobuf::Message* out_options,
              tensorflow::Tensor* out_val1,
              tensorflow::Tensor* out_val2) override;

 private:
};

class RequestPushGradHandler final : public RequestHandler {
 public:
  explicit RequestPushGradHandler(int distributed_mode)
      : RequestHandler(distributed_mode) {}
  virtual ~RequestPushGradHandler() {}
  bool Handle(Role role, int32_t role_id, uint64_t seq_id,
              const std::string& var_name,
              const ::google::protobuf::Any& options,
              const tensorflow::Tensor& val1, const tensorflow::Tensor& val2,
              ::google::protobuf::Message* out_options,
              tensorflow::Tensor* out_val1,
              tensorflow::Tensor* out_val2) override;

 private:
};

}  // namespace rpc
}  // namespace sniper
