#pragma once

#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "core/proto/meta.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace sniper {
namespace rpc {

// Source provides a way for a particular RPC implementation to provide
// received data to ParseFrom.
class Source {
 public:
  virtual ~Source() {}

  // Return the stream that contains the data to be parsed.
  // Note that this method might be invoked more than once if
  // ParseFrom needs to fall back to a more expensive parsing method.
  // Every call must return a stream pointing at the beginning of
  // the serialized TensorMessage.
  //
  // Note that a subsequent call to contents() invalidates previous
  // results of contents().
  //
  // Ownership of the returned stream is retained by the Source and
  // should not be deleted by the caller.
  virtual ::google::protobuf::io::ZeroCopyInputStream* contents() = 0;
};

class TensorResponse {
 public:
  TensorResponse() {}
  ~TensorResponse() {}

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  int Parse(Source* source);
  const tensorflow::Tensor& tensor1() const { return tensor1_; }
  const tensorflow::Tensor& tensor2() const { return tensor2_; }
  const TensorMessage& meta() const { return meta_; }

 protected:
  bool ParseTensorSubmessage(::google::protobuf::io::CodedInputStream* input,
                             TensorProto* tensor_meta,
                             tensorflow::Tensor& tensor);  // NOLINT

  bool already_used_ = false;
  tensorflow::Tensor tensor1_;
  tensorflow::Tensor tensor2_;
  TensorMessage meta_;
};

void TensorToProtoContent(const tensorflow::Tensor& src, TensorProto* proto);

};  // namespace rpc
};  // namespace sniper
