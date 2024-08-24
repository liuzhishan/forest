#include <chrono>

#include "gflags/gflags.h"
#include "core/operators/kernels/feed_queue.h"
#include "core/operators/kernels/train_config.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

DEFINE_bool(check_nan, false, "");

namespace sniper {
namespace ops {

using namespace tensorflow;

class FeedQueueOp : public ResourceOpKernel<FeedQueue> {
 public:
  explicit FeedQueueOp(OpKernelConstruction* ctx)
      : ResourceOpKernel<FeedQueue>(ctx) {}

 public:
  tensorflow::Status CreateResource(FeedQueue** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const NodeDef& ndef = def();

    std::string conf_file;
    int32_t trainer_id;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "conf_file", &conf_file));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "trainer_id", &trainer_id));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "work_mode", &work_mode));

    LOG(INFO) << "create FeedQueue in FeedQueueOp";
    *ret = new FeedQueue(conf_file, trainer_id, work_mode);
    return tensorflow::Status::OK();
  }
};

class PrefetchOp : public OpKernel {
 public:
  explicit PrefetchOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    OP_REQUIRES_OK(context, context->GetAttr("work_mode", &work_mode_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
  }
  ~PrefetchOp() {}

  void Compute(OpKernelContext* context) override {
    FeedQueue* queue = nullptr;
    OP_REQUIRES_OK(
        context, LookupResource(context, HandleFromInput(context, 0), &queue));
    FeatureColumn* feature = nullptr;
    bool over = false;
    auto ret = queue->feed(feature, &over);
    if (!ret) {
      std::cerr << "ERROR: feed fail" << std::endl;
      OP_REQUIRES(context, ret, errors::Internal("feed fail", ""));
    }

    if (over) {
      std::cerr << "INFO: queue is over" << std::endl;
      OP_REQUIRES(context, !over, errors::OutOfRange("feed queue is over", ""));
    }

    if (feature == nullptr) {
      std::cerr << "ERROR: feature is nullptr" << std::endl;
      OP_REQUIRES(context, feature != nullptr,
                  errors::Internal("feature null", ""));
    }

    context->set_output(0, feature->batchid_tensor);
    context->set_output(1, feature->label_tensor);
    context->set_output(2, feature->dense_tensor);

    auto& input_sparse = train_config_->input_sparse();
    for (size_t i = 0; i < input_sparse.size(); ++i) {
      if (FLAGS_check_nan) {
        auto flat = feature->embedding_tensor[i].flat<float>();
        auto size = feature->embedding_tensor[i].NumElements();
        for (int j = 0; j < size; ++j) {
          if (std::isnan(flat(j))) {
            SPDLOG_WARN("batch_id[{0}] lookup ret has nan",
                        feature->batchid_tensor.DebugString());
            break;
          }
        }
      }

      context->set_output(3 + i, feature->embedding_tensor[i]);
    }

    if (train_config_->debug_info().size() > 0) {
      context->set_output(3 + input_sparse.size(), feature->debug_info);
    }

    delete feature;
  }

 private:
  std::string conf_file_;
  int32_t trainer_id_;
  int32_t work_mode_;
  TrainConfig* train_config_;
};

REGISTER_KERNEL_BUILDER(Name("FeedQueue").Device(DEVICE_CPU), FeedQueueOp);
REGISTER_KERNEL_BUILDER(Name("Prefetch").Device(DEVICE_CPU), PrefetchOp);

}  // namespace ops
}  // namespace sniper
