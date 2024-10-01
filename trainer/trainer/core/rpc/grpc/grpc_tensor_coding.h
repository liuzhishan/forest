#pragma once

#include <string>

#include "grpcpp/impl/codegen/byte_buffer.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/tensor_response.h"
#include "tensorflow/core/framework/tensor.h"

namespace sniper {
namespace rpc {

void EncodeTensorMessageToByteBuffer(const TensorMessage& proto,
                                     ::grpc::ByteBuffer* result);
void EncodeTensorToByteBuffer(Role role, int32_t role_id, uint64_t seq_id,
                              const std::string& varname,
                              const ::google::protobuf::Message& options,
                              const tensorflow::Tensor& val1,
                              const tensorflow::Tensor& val2,
                              ::grpc::ByteBuffer* result);
size_t EncodeTensorToGrpcSlice(const tensorflow::Tensor& val,
                               tensorflow::uint32 tag, size_t& acc_size,
                               ::grpc::Slice& slice1, ::grpc::Slice& slice2);

void DecodeTensorFromByteBuffer(TensorResponse* response,
                                const ::grpc::ByteBuffer& buf);

}  // namespace rpc
}  // namespace sniper
