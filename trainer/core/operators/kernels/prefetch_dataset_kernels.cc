#include <chrono>
#include <string>

#include "gflags/gflags.h"
#include "core/operators/kernels/feed_queue.h"
#include "core/operators/kernels/train_config.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "core/base/semaphore.h"

DEFINE_bool(check_nan, false, "");

namespace sniper {
namespace ops {

using namespace tensorflow;

class InputPrefetchDatasetOp : public DatasetOpKernel {
 public:
  explicit InputPrefetchDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    OP_REQUIRES_OK(ctx,
                   tensorflow::data::ParseScalarArgument<std::string>(ctx, "conf_file", &conf_file_));
    OP_REQUIRES_OK(ctx, tensorflow::data::ParseScalarArgument<tensorflow::int32>(
                            ctx, "trainer_id", &trainer_id_));
    OP_REQUIRES_OK(ctx, tensorflow::data::ParseScalarArgument<tensorflow::int32>(
                            ctx, "work_mode", &work_mode_));

    *output = new Dataset(ctx, conf_file_, trainer_id_, output_types_,
                          output_shapes_, work_mode_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::string& conf_file,
            int32_t trainer_id, const DataTypeVector& output_types,
            const std::vector<tensorflow::PartialTensorShape>& output_shapes,
            int32_t work_mode)
        : DatasetBase(DatasetContext(ctx)),
          conf_file_(conf_file),
          trainer_id_(trainer_id),
          work_mode_(work_mode),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
      feed_queue_ = new FeedQueue(conf_file_, trainer_id_, work_mode_);
      feed_queue_->reset();
    }
    ~Dataset() {
      if (feed_queue_) {
        delete feed_queue_;
      }
      LOG(INFO) << "Dataset destruction";
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Prefetch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    std::string DebugString() const override {
      return "InputPrefetchDatasetOp::Dataset";
    }

   protected:
    tensorflow::Status AsGraphDefInternal(SerializationContext* ctx,
                                          DatasetGraphDefBuilder* b,
                                          Node** output) const override {
      Node* conf_file = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(conf_file_, &conf_file));
      Node* trainer_id = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(trainer_id_, &trainer_id));
      Node* work_mode = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(work_mode_, &work_mode));

      TF_RETURN_IF_ERROR(b->AddDataset(this, {conf_file, trainer_id, work_mode},
                                       output));
      return tensorflow::Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      tensorflow::Status GetNextInternal(IteratorContext* ctx,
                                         std::vector<Tensor>* out_tensors,
                                         bool* end_of_sequence) override {
        bool over = false;
        FeatureColumn* feature = nullptr;
        if (dataset()->train_config_->debug_offline()) {
          SemaphoreLoop::GetInstance().Acquire(1);
        }
        auto ret = dataset()->feed_queue_->feed(feature, &over);
        if (!ret) {
          return errors::Internal("feed fail", "");
        }

        if (over) {
          *end_of_sequence = true;
          LOG(INFO) << "GetNextInternal over, end_of_sequence";
          return tensorflow::Status::OK();
        }
        if (feature == nullptr) {
          return errors::Internal("prefetch fail, feature is null", "");
        }

        out_tensors->emplace_back(std::move(feature->batchid_tensor));
        out_tensors->emplace_back(std::move(feature->label_tensor));
        out_tensors->emplace_back(std::move(feature->dense_tensor));

        auto& input_sparse = dataset()->train_config_->input_sparse();
        for (size_t i = 0; i < input_sparse.size(); ++i) {
          auto flat = feature->embedding_tensor[i].flat<float>();
          auto size = feature->embedding_tensor[i].NumElements();
          for (int j = 0; j < size; ++j) {
            if (std::isnan(flat(j))) {
              LOG(FATAL) << "embedding lookup ret has nan, i: " << i
                         << ", j: " << j
                         << ", flat.size: " << flat.size();
            }
          }

          out_tensors->emplace_back(std::move(feature->embedding_tensor[i]));
        }

        if (dataset()->train_config_->debug_info().size() > 0) {
          out_tensors->emplace_back(std::move(feature->debug_info));
        }

        if (dataset()->train_config_->debug_offline()) {
          SemaphoreLoop::GetInstance().Release(2);
        }
        *end_of_sequence = false;
        delete feature;
        return tensorflow::Status::OK();
      }

     protected:
      tensorflow::Status SaveInternal(IteratorStateWriter* writer) override {
        return tensorflow::Status::OK();
      }

      tensorflow::Status RestoreInternal(IteratorContext* ctx,
                                         IteratorStateReader* reader) override {
        return tensorflow::Status::OK();
      }

     private:
      tensorflow::mutex mu_;
      bool run_ GUARDED_BY(mu_) = true;
    };

    std::string conf_file_;
    int32_t trainer_id_;
    int32_t work_mode_;
    const tensorflow::DataTypeVector output_types_;
    const std::vector<tensorflow::PartialTensorShape> output_shapes_;

    TrainConfig* train_config_ = nullptr;
    FeedQueue* feed_queue_ = nullptr;
  };

 private:
  std::string conf_file_;
  int32_t trainer_id_;
  int32_t work_mode_;
  tensorflow::DataTypeVector output_types_;
  std::vector<tensorflow::PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("InputPrefetchDataset").Device(DEVICE_CPU),
                        InputPrefetchDatasetOp);

}  // namespace ops
}  // namespace sniper
