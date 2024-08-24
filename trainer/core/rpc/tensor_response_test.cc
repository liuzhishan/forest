#include "core/rpc/tensor_response.h"

#include "gtest/gtest.h"
#include "core/rpc/tensor_response.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace sniper {
namespace rpc {

class StringSource : public Source {
 public:
  explicit StringSource(const std::string* s, int block_size)
      : s_(s), stream_(nullptr), block_size_(block_size) {}
  ~StringSource() override { DeleteStream(); }

  ::google::protobuf::io::ZeroCopyInputStream* contents() override {
    DeleteStream();
    stream_ = new (&space_)::google::protobuf::io::ArrayInputStream(
        s_->data(), s_->size(), block_size_);
    return stream_;
  }

  void DeleteStream() {
    if (stream_) {
      stream_->~ArrayInputStream();
    }
  }

 private:
  const std::string* s_;
  ::google::protobuf::io::ArrayInputStream* stream_;
  char space_[sizeof(::google::protobuf::io::ArrayInputStream)];
  int block_size_;
};

void Validate(const tensorflow::Tensor& src1, const tensorflow::Tensor& src2,
              Role role, uint32_t role_id, uint64_t seq_id,
              const std::string& varname) {
  TensorMessage proto;
  proto.set_role(role);
  proto.set_role_id(role_id);
  proto.set_seq_id(seq_id);
  proto.set_varname(varname);

  TensorToProtoContent(src1, proto.mutable_tensor1());
  TensorToProtoContent(src2, proto.mutable_tensor2());

  std::string encoded;
  proto.AppendToString(&encoded);

  StringSource source(&encoded, 1024);

  TensorResponse response;
  for (int i = 0; i < 2; i++) {  // Twice so we exercise reuse of "response"
    int ret = response.Parse(&source);
    EXPECT_EQ(ret, 0);

    const TensorMessage& meta = response.meta();
    EXPECT_EQ(meta.role(), role);
    EXPECT_EQ(meta.role_id(), role_id);
    EXPECT_EQ(meta.seq_id(), seq_id);
    EXPECT_EQ(meta.varname(), varname);

    const tensorflow::Tensor& result1 = response.tensor1();
    EXPECT_EQ(result1.dtype(), src1.dtype());
    EXPECT_EQ(result1.shape().DebugString(), src1.shape().DebugString());
    EXPECT_EQ(result1.DebugString(), src1.DebugString());

    const tensorflow::Tensor& result2 = response.tensor2();
    EXPECT_EQ(result2.dtype(), src2.dtype());
    EXPECT_EQ(result2.shape().DebugString(), src2.shape().DebugString());
    EXPECT_EQ(result2.DebugString(), src2.DebugString());
  }
}

template <typename T>
void DoTest(tensorflow::DataType dt) {
  tensorflow::gtl::InlinedVector<T, 4> v;
  for (int elems = 0; elems <= 10000; elems++) {
    if (elems < 100 || (elems % 1000 == 0)) {
      tensorflow::Tensor a(
          dt, tensorflow::TensorShape({1, static_cast<int64_t>(v.size())}));
      auto flat = a.flat<T>();
      if (flat.size() > 0) {
        std::copy_n(v.data(), v.size(), flat.data());
      }
      Validate(a, a, ROLE_TRAINER, 0, 123, "test_var");
    }
    v.push_back(static_cast<T>(elems));
  }
}

TEST(TensorResponseTest, Simple) {
  DoTest<float>(tensorflow::DT_FLOAT);
  DoTest<tensorflow::int64>(tensorflow::DT_INT64);
}

}  // namespace rpc
}  // namespace sniper
