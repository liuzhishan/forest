#include "trainer/core/rpc/tensor_response.h"

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "trainer/core/proto/meta.pb.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/types.pb.h"

namespace sniper {
namespace rpc {

// Define some helper routines for decoding protocol buffer wire format data
namespace {
// We only need some of the wiretype values for this code
enum WireType {
  WIRETYPE_VARINT = 0,
  WIRETYPE_LENGTH_DELIMITED = 2,
};
inline int GetTagFieldNumber(uint32_t tag) { return tag >> 3; }
inline WireType GetTagWireType(uint32_t tag) {
  return static_cast<WireType>(tag & 0x7);
}

bool ReadVarintSizeAsInt(::google::protobuf::io::CodedInputStream* input,
                         int* result) {
  uint64_t v;
  if (input->ReadVarint64(&v) && v <= static_cast<uint64_t>(INT_MAX)) {
    *result = static_cast<int>(v);
    return true;
  } else {
    return false;
  }
}

bool ReadNestedMessage(::google::protobuf::io::CodedInputStream* input,
                       ::google::protobuf::Message* value) {
  int length;
  if (!ReadVarintSizeAsInt(input, &length)) return false;
  std::pair<::google::protobuf::io::CodedInputStream::Limit, int> p =
      input->IncrementRecursionDepthAndPushLimit(length);
  if (p.second < 0 || !value->MergePartialFromCodedStream(input)) return false;
  // Make sure that parsing stopped when the limit was hit, not at an endgroup
  // tag.
  return input->DecrementRecursionDepthAndPopLimit(p.first);
}

}  // namespace

int TensorResponse::Parse(Source* source) {
  if (already_used_) {
    meta_.Clear();
    tensor1_ = tensorflow::Tensor();
    tensor2_ = tensorflow::Tensor();
  }
  already_used_ = true;

  ::google::protobuf::io::CodedInputStream input(source->contents());
  input.SetTotalBytesLimit(INT_MAX, INT_MAX);  // Unlimited
  while (true) {
    auto p = input.ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      if (tag == 0) {
        return 0;
      } else {
        return -2;
      }
    }
    switch (tag) {
      case TensorMessage::kRoleFieldNumber: {
        uint32_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return -3;
        }
        meta_.set_role(static_cast<Role>(static_cast<int>(v)));
        break;
      }
      case TensorMessage::kRoleIdFieldNumber: {
        uint32_t v;
        std::string var_name;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return -4;
        }
        meta_.set_role_id(v);
        break;
      }
      case TensorMessage::kSeqIdFieldNumber: {
        uint64_t v;
        std::string var_name;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return -5;
        }
        meta_.set_seq_id(v);
        break;
      }
      case TensorMessage::kVarnameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return -6;
        }
        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return -7;
        }
        meta_.set_varname(temp);
        break;
      }
      case TensorMessage::kOptionsFieldNumber: {
        if ((wt != WIRETYPE_LENGTH_DELIMITED) ||
            !ReadNestedMessage(&input, meta_.mutable_options())) {
          return -8;
        }
        break;
      }
      case TensorMessage::kTensor1FieldNumber: {
        if (wt != WIRETYPE_LENGTH_DELIMITED) return -9;

        int length;
        if (!ReadVarintSizeAsInt(&input, &length)) return -10;
        std::pair<::google::protobuf::io::CodedInputStream::Limit, int> p =
            input.IncrementRecursionDepthAndPushLimit(length);
        if (p.second < 0 ||
            !ParseTensorSubmessage(&input, meta_.mutable_tensor1(), tensor1_)) {
          return -11;
        }
        if (!input.DecrementRecursionDepthAndPopLimit(p.first)) {
          return -12;
        }
        break;
      }
      case TensorMessage::kTensor2FieldNumber: {
        if (wt != WIRETYPE_LENGTH_DELIMITED) return false;

        int length;
        if (!ReadVarintSizeAsInt(&input, &length)) return false;
        std::pair<::google::protobuf::io::CodedInputStream::Limit, int> p =
            input.IncrementRecursionDepthAndPushLimit(length);
        if (p.second < 0 ||
            !ParseTensorSubmessage(&input, meta_.mutable_tensor2(), tensor2_)) {
          return -13;
        }
        if (!input.DecrementRecursionDepthAndPopLimit(p.first)) {
          return -14;
        }
        break;
      }
      default: {
        // Unknown tag, so don't handle we can't handle on the fast path
        return -1;
      }
    }
  }

  return 0;
}

bool TensorResponse::ParseTensorSubmessage(
    ::google::protobuf::io::CodedInputStream* input, TensorProto* tensor_meta,
    tensorflow::Tensor& tensor) {
  bool seen_tensor_content = false;
  while (true) {
    auto p = input->ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      bool ok = (tag == 0);
      if (ok && !seen_tensor_content) {
        // No tensor content: could be because it's a zero-length tensor
        std::vector<tensorflow::int64> dims;
        for (int i = 0; i < tensor_meta->tensor_shape().dim_size(); ++i) {
          dims.push_back(tensor_meta->tensor_shape().dim(i));
        }
        tensorflow::TensorShape shape(absl::MakeSpan(dims));
        tensorflow::Tensor t(
            static_cast<tensorflow::DataType>(tensor_meta->dtype()), shape);
        tensor = std::move(t);
      }
      return ok;
    }
    switch (tag) {
      case TensorProto::kDtypeFieldNumber: {
        uint32_t v;
        if ((wt != WIRETYPE_VARINT) || !input->ReadVarint32(&v)) return false;
        if (seen_tensor_content) return false;
        tensor_meta->set_dtype(static_cast<DataType>(static_cast<int>(v)));
        break;
      }
      case TensorProto::kTensorShapeFieldNumber: {
        if ((wt != WIRETYPE_LENGTH_DELIMITED) ||
            !ReadNestedMessage(input, tensor_meta->mutable_tensor_shape()))
          return false;
        if (seen_tensor_content) return false;
        break;
      }
      case TensorProto::kTensorContentFieldNumber: {
        // If we haven't seen the dtype and tensor_shape data first, we can't
        // deal with this in the fast path.
        if (seen_tensor_content) return false;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !tensor_meta->has_tensor_shape()) {
          return false;
        }
        int num_bytes;
        if (!ReadVarintSizeAsInt(input, &num_bytes)) return false;
        seen_tensor_content = true;

        std::vector<tensorflow::int64> dims;
        for (int i = 0; i < tensor_meta->tensor_shape().dim_size(); ++i) {
          dims.push_back(tensor_meta->tensor_shape().dim(i));
        }
        tensorflow::TensorShape shape(absl::MakeSpan(dims));

        tensorflow::Tensor t(
            static_cast<tensorflow::DataType>(tensor_meta->dtype()), shape);
        auto buf = t.tensor_data();

        if (static_cast<size_t>(num_bytes) != buf.size()) {
          return false;
        }
        // TODO(jeff,sanjay): Figure out a way to avoid this copy if
        // the underlying ZeroCopyInputStream data is properly aligned
        // and compatible with what allocator_ wants.
        if (!input->ReadRaw(const_cast<char*>(buf.data()), num_bytes)) {
          return false;
        }
        tensor = std::move(t);
        break;
      }
      default: {
        // Some other tag our fast path code is not prepared to handle.
        // return false.
        return false;
      }
    }
  }
}  // namespace rpc

void TensorToProtoContent(const tensorflow::Tensor& src, TensorProto* proto) {
  proto->Clear();
  proto->set_dtype(static_cast<DataType>(src.dtype()));

  for (int i = 0; i < src.shape().dims(); ++i) {
    proto->mutable_tensor_shape()->add_dim(src.shape().dim_size(i));
  }

  auto buf = tensorflow::DMAHelper::buffer(&src);
  if (buf) {
    proto->mutable_tensor_content()->assign(buf->base<const char>(),
                                            buf->size());
  }
}

}  // namespace rpc
}  // namespace sniper
