#include "core/rpc/request_handler_impl.h"

#include <iostream>
#include <string>
#include <vector>

#include "core/util/monitor/run_status.h"
#include "spdlog/spdlog.h"
#include "tensorflow/core/framework/bfloat16.h"

namespace sniper {
namespace rpc {

bool RequestLookupHandler::Handle(
    Role role, int32_t role_id, uint64_t seq_id, const std::string& varname,
    const ::google::protobuf::Any& options, const tensorflow::Tensor& val1,
    const tensorflow::Tensor& val2, ::google::protobuf::Message* out_options,
    tensorflow::Tensor* out_val1, tensorflow::Tensor* out_val2) {
  SPDLOG_INFO(
      "RequestLookupHandler, Role: {0}, role_id: {1}, seq_id: {2}, varname: "
      "{3}, option: {4}, var1: {5}, var2: {6}",
      role, role_id, seq_id, varname, options.DebugString(), val1.DebugString(),
      val2.DebugString());

  auto start = std::chrono::steady_clock::now();
  tensorflow::Tensor t(tensorflow::DT_FLOAT, {1024, 16});

  tensorflow::Tensor bfloat16_out_val1(tensorflow::DT_BFLOAT16, t.shape());
  tensorflow::FloatToBFloat16(
      t.flat<float>().data(),
      bfloat16_out_val1.flat<tensorflow::bfloat16>().data(), t.NumElements());
  *out_val1 = bfloat16_out_val1;

  auto end = std::chrono::steady_clock::now();
  monitor::RunStatus::Instance()->PushTime(
      monitor::kPsEmbeddingLookup,
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());

  return true;
}

bool RequestPushGradHandler::Handle(
    Role role, int32_t role_id, uint64_t seq_id, const std::string& varname,
    const ::google::protobuf::Any& options, const tensorflow::Tensor& val1,
    const tensorflow::Tensor& val2, ::google::protobuf::Message* out_options,
    tensorflow::Tensor* out_val1, tensorflow::Tensor* out_val2) {
  SPDLOG_INFO(
      "RequestPushGradHandler, Role: {0}, role_id: {1}, seq_id: {2}, varname: "
      "{3}, option: {4}, var1: {5}, var2: {6}",
      role, role_id, seq_id, varname, options.DebugString(), val1.DebugString(),
      val2.DebugString());

  auto start = std::chrono::steady_clock::now();

  tensorflow::Tensor t(tensorflow::DT_FLOAT, val1.shape());
  tensorflow::BFloat16ToFloat(val1.flat<tensorflow::bfloat16>().data(),
                              t.flat<float>().data(), t.NumElements());

  auto end = std::chrono::steady_clock::now();
  monitor::RunStatus::Instance()->PushTime(
      monitor::kPsPushGrad,
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());

  return true;
}

}  // namespace rpc
}  // namespace sniper
