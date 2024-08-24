#include "core/rpc/grpc/grpc_tensor_coding.h"

#include "grpcpp/support/byte_buffer.h"
#include "grpcpp/support/slice.h"
#include "core/rpc/grpc/grpc_bytebuffer_stream.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/proto_encode_helper.h"

namespace sniper {
namespace rpc {

void EncodeTensorMessageToByteBuffer(const TensorMessage& proto,
                                     ::grpc::ByteBuffer* result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(const_cast<tensorflow::uint8*>(
      reinterpret_cast<const tensorflow::uint8*>(slice.begin())));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}

static int VarLengthEncodingSize(tensorflow::uint32 tag, size_t bytes) {
  return tensorflow::core::VarintLength(tag << 3) +
         tensorflow::core::VarintLength(bytes) + bytes;
}

// Returns an upper bound in bytes of the protocol buffer encoding of
// the "skeleton" of "val" (all the data needed for dtype and the shape,
// but not the actual contents of "val").
static int SkeletonEncodingSizeUpperBound(const tensorflow::Tensor& val) {
  static const int kVarintMax64 = 10;  // Max length of varint64 encoding
  const int ndims = val.shape().dims();
  return (2 * kVarintMax64) +           // dtype
         (ndims * (4 * kVarintMax64));  // Shape: 4 varints per dim
}

// Encode the skeleton for "val" (the encoded TensorProto contents
// (dtype and shape, but not the actual data) into "*e".  The backing
// store for "*e" must be of appropriate size to hold this encoding.
static void EncodeSkeleton(const tensorflow::Tensor& val,
                           tensorflow::io::ProtoEncodeHelper* e) {
  // Encode val.dtype()
  e->WriteUint64(TensorProto::kDtypeFieldNumber, val.dtype());

  // Compute length of val.shape() proto encoding
  const int ndims = val.shape().dims();
  int tensor_shape_bytes = 0;
  for (int d = 0; d < ndims; d++) {
    tensorflow::int64 dim_size = val.shape().dim_size(d);
    tensor_shape_bytes += 1 +  // TensorShapeProto::kDimFieldNumber
                          tensorflow::core::VarintLength(dim_size);
  }

  if (tensor_shape_bytes > 0) {
    e->WriteVarlengthBeginning(TensorProto::kTensorShapeFieldNumber,
                               tensor_shape_bytes);
    // Encode val.shape()
    for (int d = 0; d < ndims; d++) {
      tensorflow::int64 dim_size = val.shape().dim_size(d);
      e->WriteUint64(TensorShapeProto::kDimFieldNumber, dim_size);
    }
  }
}

void EncodeTensorToByteBuffer(Role role, int32_t role_id, uint64_t seq_id,
                              const std::string& varname,
                              const ::google::protobuf::Message& options,
                              const tensorflow::Tensor& val1,
                              const tensorflow::Tensor& val2,
                              ::grpc::ByteBuffer* result) {
  TensorMessage message;
  message.set_role(role);
  message.set_role_id(role_id);
  message.set_seq_id(seq_id);
  message.set_varname(varname);
  message.mutable_options()->PackFrom(options);

  std::string header;
  message.AppendToString(&header);

  size_t expected_size = header.size();

  // Now allocate memory and put into the ByteBuffer
  ::grpc::Slice slices[5];  // meta, tensor1, tensor2

  // meta
  int num_slices = 0;
  {
    tensorflow::gtl::InlinedVector<char, 4096> space(
        header.size());  // we need 4k to avoid reallocate on heap
    tensorflow::io::ProtoEncodeHelper e(space.data(), space.size());
    e.WriteRawBytes(header);

    slices[0] = ::grpc::Slice(e.size());
    memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
    num_slices += 1;
  }

  auto ret = EncodeTensorToGrpcSlice(val1, TensorMessage::kTensor1FieldNumber,
                                     expected_size, slices[1], slices[2]);
  num_slices += ret;
  if (ret == 1) {
    ret = EncodeTensorToGrpcSlice(val2, TensorMessage::kTensor2FieldNumber,
                                  expected_size, slices[2], slices[3]);
  } else {
    ret = EncodeTensorToGrpcSlice(val2, TensorMessage::kTensor2FieldNumber,
                                  expected_size, slices[3], slices[4]);
  }
  num_slices += ret;

  size_t total_bytes = 0;
  for (int i = 0; i < num_slices; i++) {
    total_bytes += slices[i].size();
  }
  CHECK_EQ(total_bytes, expected_size);

  ::grpc::ByteBuffer tmp(&slices[0], num_slices);
  result->Swap(&tmp);
}

size_t EncodeTensorToGrpcSlice(const tensorflow::Tensor& val,
                               // NOLINTNEXTLINE
                               tensorflow::uint32 tag, size_t& acc_size,
                               // NOLINTNEXTLINE
                               ::grpc::Slice& slice1, ::grpc::Slice& slice2) {
  const int kLargeTensorBytes = 1024;
  tensorflow::gtl::InlinedVector<char, 128> skeleton(
      SkeletonEncodingSizeUpperBound(val));
  tensorflow::io::ProtoEncodeHelper e_skeleton(skeleton.data(),
                                               skeleton.size());
  EncodeSkeleton(val, &e_skeleton);
  tensorflow::StringPiece tdata = val.tensor_data();

  // If "share_tensor_slice_memory == false", we copy the tensor data to
  // the end of the buffer we are preparing that holds the rest of the
  // TensorMessage protocol buffer.
  //
  // If "share_tensor_slice_memory == true", we arrange to share the
  // backing store of the data by creating a slice that also points to the
  // backing store, with appropriate reference counts to keep the
  // backing store alive as needed.
  //
  // We enable this behavior if the tensor is large.
  bool share_tensor_slice_memory = (tdata.size() > kLargeTensorBytes);

  tensorflow::uint32 overall_tensor_proto_bytesize =
      (e_skeleton.size() +
       VarLengthEncodingSize(TensorProto::kTensorContentFieldNumber,
                             tdata.size()));
  size_t t_field_size =
      VarLengthEncodingSize(tag, overall_tensor_proto_bytesize);
  acc_size += t_field_size;

  size_t encoder_size = t_field_size - tdata.size();

  tensorflow::gtl::InlinedVector<char, 1024> space(encoder_size);
  tensorflow::io::ProtoEncodeHelper e(space.data(), space.size());

  e.WriteVarlengthBeginning(tag, overall_tensor_proto_bytesize);
  e.WriteRawBytes(
      tensorflow::StringPiece(e_skeleton.data(), e_skeleton.size()));
  e.WriteVarlengthBeginning(TensorProto::kTensorContentFieldNumber,
                            tdata.size());

  size_t slice_len = e.size() + (share_tensor_slice_memory ? 0 : tdata.size());
  slice1 = ::grpc::Slice(slice_len);
  memcpy(const_cast<uint8_t*>(slice1.begin()), e.data(), e.size());
  if (!share_tensor_slice_memory) {
    memcpy(const_cast<uint8_t*>(slice1.begin()) + e.size(), tdata.data(),
           tdata.size());
    return 1;
  } else {
    // (E) Encode tensor data, but by sharing backing store
    const tensorflow::TensorBuffer* buf = tensorflow::DMAHelper::buffer(&val);
    buf->Ref();
    slice2 = ::grpc::Slice(
        const_cast<void*>(static_cast<const void*>(tdata.data())), tdata.size(),
        [](void* backing) {
          static_cast<tensorflow::TensorBuffer*>(backing)->Unref();
        },
        const_cast<tensorflow::TensorBuffer*>(buf));
    return 2;
  }
}  // namespace rpc

void DecodeTensorFromByteBuffer(TensorResponse* response,
                                const ::grpc::ByteBuffer& buf) {
  GrpcByteBufferSource source;
  source.Init(buf);
  GrpcByteBufferSourceWrapper r(&source);
  CHECK_EQ(response->Parse(&r), 0);
}

}  // namespace rpc
}  // namespace sniper
