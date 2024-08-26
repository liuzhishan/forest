#include "trainer/core/rpc/grpc/grpc_tensor_coding.h"

#include <vector>

#include "grpcpp/support/byte_buffer.h"
#include "grpcpp/support/slice.h"
#include "gtest/gtest.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_bytebuffer_stream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace sniper {
namespace rpc {

void Validate(const tensorflow::Tensor& t) {
  // Check by encoding to a ByteBuffer
  ::grpc::ByteBuffer buf;
  PushGradOption option;
  option.set_learning_rate(0.123);
  EncodeTensorToByteBuffer(ROLE_TRAINER, 0, 123, "test_var", option, t, t, &buf);

  // Make a string
  std::vector<::grpc::Slice> slices;
  (void)buf.Dump(&slices);
  std::string tmp;
  for (const auto& s : slices) {
    tmp.append(reinterpret_cast<const char*>(s.begin()), s.size());
  }

  TensorMessage message;
  EXPECT_TRUE(message.ParseFromString(tmp));
  EXPECT_EQ(message.role(), ROLE_TRAINER);
  EXPECT_EQ(message.role_id(), 0);
  EXPECT_EQ(message.seq_id(), 123);
  EXPECT_EQ(message.varname(), "test_var");

  GrpcByteSource source(&buf);
  TensorResponse response;
  auto ret = response.Parse(&source);
  EXPECT_EQ(ret, 0);

  auto& result_tensor1 = response.tensor1();
  EXPECT_EQ(t.dtype(), result_tensor1.dtype());
  EXPECT_EQ(t.shape().DebugString(), result_tensor1.shape().DebugString());
  EXPECT_EQ(t.DebugString(), result_tensor1.DebugString());

  auto& result_tensor2 = response.tensor2();
  EXPECT_EQ(t.dtype(), result_tensor2.dtype());
  EXPECT_EQ(t.shape().DebugString(), result_tensor2.shape().DebugString());
  EXPECT_EQ(t.DebugString(), result_tensor2.DebugString());

  PushGradOption result_option;
  EXPECT_TRUE(message.options().UnpackTo(&result_option));
  EXPECT_EQ(option.learning_rate(), result_option.learning_rate());
}

template <typename T>
void DoTest(tensorflow::DataType dt) {
  tensorflow::gtl::InlinedVector<T, 4> v;
  for (int elems = 0; elems <= 10000; elems++) {
    if (elems < 100 || (elems % 1000 == 0)) {
      tensorflow::Tensor a(dt,
                           tensorflow::TensorShape(
                               {1, static_cast<tensorflow::int64>(v.size())}));
      auto flat = a.flat<T>();
      if (flat.size() > 0) {
        std::copy_n(v.data(), v.size(), flat.data());
      }
      Validate(a);
    }
    v.push_back(static_cast<T>(elems));
  }
}

TEST(GrpcTensorCodingTest, Simple) {
  DoTest<float>(tensorflow::DT_FLOAT);
  DoTest<tensorflow::int32>(tensorflow::DT_INT32);
}

}  // namespace rpc
}  // namespace sniper
