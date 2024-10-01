#include "glog/logging.h"
#include "trainer/core/operators/kernels/train_config.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_client.h"
#include "trainer/core/util/monitor/run_status.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "trainer/core/util/placement/auto_shard.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class RestoreOp : public OpKernel {
 public:
  explicit RestoreOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";

    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));

    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);

    AutoShard::instance().add_placement(train_config_->placement());

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("shard_idx", &shard_idx_));
    OP_REQUIRES_OK(context, context->GetAttr("shard_num", &shard_num_));
    OP_REQUIRES_OK(context, context->GetAttr("shard_nfs_weight_paths",
                                             &shard_nfs_weight_paths_));
    OP_REQUIRES_OK(context, context->GetAttr("shard_nfs_adagrad_paths",
                                             &shard_nfs_adagrad_paths_));
  }

  ~RestoreOp() {}

  void Compute(OpKernelContext* context) override {
    auto start = std::chrono::steady_clock::now();

    std::vector<std::string> eps;
    bool is_sparse = true;

    if (varname_.rfind("embedding_", 0) == 0) {
      eps = train_config_->placement()->GetEmbPlacement(varname_);

      is_sparse = true;
    } else {
      auto ep = train_config_->placement()->GetDensePlacement(varname_);
      eps.push_back(ep);

      is_sparse = false;
    }

    OP_REQUIRES(context,
                (shard_idx_ < eps.size()) && (shard_num_ == eps.size()),
                errors::InvalidArgument(
                    varname_ + " placement shard num not match shard_idx"));
    OP_REQUIRES(
        context,
        shard_nfs_weight_paths_.size() == shard_nfs_adagrad_paths_.size(),
        errors::InvalidArgument(
            " shard_nfs_adagrad_paths_size !=  shard_nfs_adagrad_paths_size"));

    auto ep = eps[shard_idx_];

    int32_t total_iteration = 180;

    bool restore_finish = false;
    bool restore_error = false;

    RestoreOption option;

    if (is_sparse) {
      option.set_variable_type(VariableType::VAR_EMBEDDING);
    } else {
      option.set_variable_type(VariableType::VAR_DENSE);
    }

    option.set_shard_idx(shard_idx_);
    option.set_shard_num(shard_num_);

    for (size_t i = 0; i < shard_nfs_weight_paths_.size(); ++i) {
      option.add_nfs_weight_path(shard_nfs_weight_paths_[i]);
      option.add_nfs_adagrad_path(shard_nfs_adagrad_paths_[i]);
    }
    while (!restore_finish && !restore_error && (--total_iteration > 0)) {
      rpc::TensorResponse response;
      auto h = rpc_client_->RestoreAsync(ep, varname_, option, &response);
      if (!h->Wait()) {
        LOG(WARNING) << "restore var[" << varname_ << "] on ps[" << h->ep()
                     << "] fail. errmsg=" << h->errmsg();
      } else {
        RestoreOption ret_option;
        response.meta().options().UnpackTo(&ret_option);
        if (!ret_option.errmsg().empty()) {
          LOG(WARNING) << "Trainer restore fail. trainer_id: " << trainer_id_
                       << ", ep: " << ep
                       << ", errmsg: " << option.errmsg();
          restore_error = true;
        } else {
          if (ret_option.finish()) {
            restore_finish = true;
          }
        }
        OP_REQUIRES(context, restore_error == false,
                    errors::InvalidArgument("restore fail"));
      }
      std::this_thread::sleep_for(std::chrono::seconds(30));
    }

    OP_REQUIRES(context, restore_finish == true,
                errors::InvalidArgument("restore timeout, ",
                                        "var: " + varname_ + ", shard_idx: " +
                                            std::to_string(shard_idx_) +
                                            ", ps: " + ep));

    auto end = std::chrono::steady_clock::now();
    monitor::RunStatus::Instance()->PushTime(
        monitor::kOpsRestore,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());

    LOG(INFO) << "Trainer restore var success, trainer_id: " << trainer_id_
              << ", varname: " << varname_
              << ", shard_idx: " << shard_idx_
              << ", ps: " << ep
              << ", cost: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::string varname_ = "";
  int32_t shard_idx_;
  int32_t shard_num_;
  std::vector<std::string> shard_nfs_weight_paths_;
  std::vector<std::string> shard_nfs_adagrad_paths_;
};

REGISTER_KERNEL_BUILDER(Name("SniperRestore").Device(DEVICE_CPU), RestoreOp);

}  // namespace ops
}  // namespace sniper
