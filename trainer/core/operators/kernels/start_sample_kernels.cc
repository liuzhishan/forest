#include "core/operators/kernels/train_config.h"
#include "core/proto/meta.pb.h"
#include "core/rpc/grpc/grpc_client.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class StartSampleOp : public OpKernel {
 public:
  explicit StartSampleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("parallel", &parallel_));
    OP_REQUIRES_OK(context, context->GetAttr("src", &src_));
    OP_REQUIRES_OK(context, context->GetAttr("file_list", &file_list_));
    OP_REQUIRES_OK(context, context->GetAttr("work_mode", &work_mode_));

    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ =
        rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);  // worker_id
  }
  ~StartSampleOp() {}

  void Compute(OpKernelContext* ctx) override {
    StartSampleOption option;
    option.set_src(static_cast<klearn::Src>(src_));
    option.set_parallel(parallel_);
    option.set_role(ROLE_HUB);
    option.set_work_mode(static_cast<klearn::WorkMode>(work_mode_));
    TrainConfig::ConvertConfigToPb(train_config_, option);

    LOG(INFO) << "enable_format_opt: " << option.enable_format_opt()
              << ", work_mode: " << work_mode_
              << ", option.work_mode(): " << option.work_mode()
              << ", hash_type: " << option.hash_type();

    auto& hub_eps = train_config_->hub_eps();
    std::vector<std::vector<std::string>> hub_file_list_;
    hub_file_list_.resize(hub_eps.size());
    for (size_t i = 0; i < file_list_.size(); ++i) {
      hub_file_list_[i % hub_eps.size()].push_back(file_list_[i]);
    }

    int32_t btq_topic_num = train_config_->btq_topic_num();
    assert(btq_topic_num >= hub_eps.size());
    std::vector<std::pair<int32_t, int32_t>> btq_shards;
    if (btq_topic_num > 0) {
      LOG(INFO) << "Start Btq Stream ,partition size : " << btq_topic_num;
      int32_t each_shard_num = btq_topic_num / hub_eps.size();
      each_shard_num = each_shard_num == 0 ? 1 : each_shard_num;

      int32_t last_end = 0;
      for (size_t i = 0; i < hub_eps.size() - 1; i++) {
        btq_shards.emplace_back(last_end, last_end + each_shard_num - 1);
        btq_topic_num -= each_shard_num;
        last_end = last_end + each_shard_num;
      }
      btq_shards.emplace_back(last_end, last_end + btq_topic_num - 1);
    }

    std::vector<rpc::RpcHandlePtr> hdls;
    for (size_t i = 0; i < hub_eps.size(); ++i) {
      option.set_hub_idx(i);

      if (btq_topic_num > 0) {
        option.set_btq_logic_topic_start(btq_shards[i].first);
        option.set_btq_logic_topic_end(btq_shards[i].second);
      }

      *(option.mutable_hdfs_src()->mutable_file_list()) = {
          hub_file_list_[i].begin(), hub_file_list_[i].end()};
      auto h = rpc_client_->StartSampleAsync(hub_eps[i], option);
      hdls.push_back(h);
    }
    for (auto& h : hdls) {
      if (!h->Wait()) {
        LOG(WARNING) << "start sample at hub[" << h->ep()
                     << "] fail. error=" << h->errmsg();
        return;
      } else {
        LOG(WARNING) << "start sample at hub[" << h->ep() << "] success.";
      }
    }
    LOG(INFO) << "start sample success.";
  }

 private:
  std::string conf_file_;
  int32_t trainer_id_;
  TrainConfig* train_config_;

  tensorflow::int32 parallel_;
  tensorflow::int32 src_;
  std::vector<std::string> file_list_;
  int work_mode_ = 0;

  rpc::RPCClient* rpc_client_;
};

REGISTER_KERNEL_BUILDER(Name("StartSample").Device(DEVICE_CPU), StartSampleOp);

}  // namespace ops
}  // namespace sniper
