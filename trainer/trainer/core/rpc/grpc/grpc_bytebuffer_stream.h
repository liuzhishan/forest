#pragma once

#include <vector>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "grpc++/grpc++.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "trainer/core/rpc/tensor_response.h"

namespace grpc {

/// A ZeroCopyInputStream that reads from grpc_byte_buffer
class GrpcBufferReader final
    : public ::google::protobuf::io::ZeroCopyInputStream {
  typedef void (CoreCodegenInterface::*OldReaderInitAPI)(
      grpc_byte_buffer_reader* reader, grpc_byte_buffer* buffer);
  typedef int (CoreCodegenInterface::*NewReaderInitAPI)(
      grpc_byte_buffer_reader* reader, grpc_byte_buffer* buffer);
  void ReaderInit(OldReaderInitAPI ptr, grpc_byte_buffer_reader* reader,
                  grpc_byte_buffer* buffer) {
    (g_core_codegen_interface->*ptr)(reader, buffer);
  }
  void ReaderInit(NewReaderInitAPI ptr, grpc_byte_buffer_reader* reader,
                  grpc_byte_buffer* buffer) {
    int result = (g_core_codegen_interface->*ptr)(reader, buffer);
    (void)result;
  }

 public:
  explicit GrpcBufferReader(grpc_byte_buffer* buffer)
      : byte_count_(0), backup_count_(0) {
    ReaderInit(&CoreCodegenInterface::grpc_byte_buffer_reader_init, &reader_,
               buffer);
  }
  ~GrpcBufferReader() override {
    g_core_codegen_interface->grpc_byte_buffer_reader_destroy(&reader_);
  }

  bool Next(const void** data, int* size) override {
    if (backup_count_ > 0) {
      *data = GRPC_SLICE_START_PTR(slice_) + GRPC_SLICE_LENGTH(slice_) -
              backup_count_;
      GPR_CODEGEN_ASSERT(backup_count_ <= INT_MAX);
      *size = static_cast<int>(backup_count_);
      backup_count_ = 0;
      return true;
    }
    if (!g_core_codegen_interface->grpc_byte_buffer_reader_next(&reader_,
                                                                &slice_)) {
      return false;
    }
    g_core_codegen_interface->grpc_slice_unref(slice_);
    *data = GRPC_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GRPC_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = static_cast<int>(GRPC_SLICE_LENGTH(slice_));
    return true;
  }

  void BackUp(int count) override { backup_count_ = count; }

  bool Skip(int count) override {
    const void* data;
    int size;
    while (Next(&data, &size)) {
      if (size >= count) {
        BackUp(size - count);
        return true;
      }
      count -= size;
    }

    return false;
  }

  ::google::protobuf::int64 ByteCount() const override {
    return byte_count_ - backup_count_;
  }

 private:
  int64_t byte_count_;
  int64_t backup_count_;
  grpc_byte_buffer_reader reader_;
  grpc_slice slice_;
};

};  // namespace grpc

namespace sniper {
namespace rpc {

/// A ZeroCopyInputStream that reads from a grpc::ByteBuffer.
class GrpcByteBufferSource
    : public ::google::protobuf::io::ZeroCopyInputStream {
 public:
  GrpcByteBufferSource();
  bool Init(const ::grpc::ByteBuffer& src);  // Can be called multiple times.
  bool Next(const void** data, int* size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  ::google::protobuf::int64 ByteCount() const override;

 private:
  std::vector<::grpc::Slice> slices_;

  // Current slice index.
  size_t cur_;

  // Number of bytes in slices_[cur_] left to yield.
  int left_;

  // Address of next byte in slices_[cur_] to yield.
  const char* ptr_;

  ::google::protobuf::int64 byte_count_;
};

class GrpcByteBufferSourceWrapper : public Source {
 public:
  explicit GrpcByteBufferSourceWrapper(GrpcByteBufferSource* source)
      : source_(source) {}
  ::google::protobuf::io::ZeroCopyInputStream* contents() override {
    return source_;
  }

 private:
  GrpcByteBufferSource* source_;
};

class GrpcByteSource : public Source {
 public:
  explicit GrpcByteSource(grpc_byte_buffer* buffer) : buffer_(buffer) {}
  ~GrpcByteSource() override { DeleteStream(); }

  typedef ::grpc::GrpcBufferReader Reader;

  ::google::protobuf::io::ZeroCopyInputStream* contents() override {
    DeleteStream();
    stream_ = new (&space_) Reader(buffer_);
    return stream_;
  }

 private:
  void DeleteStream() {
    if (stream_) {
      stream_->~Reader();
    }
  }

  grpc_byte_buffer* buffer_;
  Reader* stream_ = nullptr;

  char space_[sizeof(Reader)];
};

}  // namespace rpc
}  // namespace sniper
