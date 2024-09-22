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

class SaveOp : public OpKernel {
 public:
  explicit SaveOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id

    AutoShard::instance().add_placement(train_config_->placement());

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("type", &type_));
    OP_REQUIRES_OK(context, context->GetAttr("target", &target_));
    OP_REQUIRES_OK(context, context->GetAttr("nfs_path", &nfs_path_));
    OP_REQUIRES_OK(context, context->GetAttr("queue", &queue_));
  }
  ~SaveOp() {}

  void Compute(OpKernelContext* context) override {
    const Tensor& version = context->input(0);
    int64_t version_ = (version.scalar<int64_t>())();

    auto emb_tables = train_config_->emb_tables();
    std::vector<std::string> eps;

    bool is_sparse = true;

    if (emb_tables.find(varname_) != emb_tables.end()) {
      eps = train_config_->placement()->GetEmbPlacement(varname_);
      is_sparse = true;
    } else {
      auto ep = train_config_->placement()->GetDensePlacement(varname_);
      eps.push_back(ep);
      is_sparse = false;
    }

    auto start = std::chrono::steady_clock::now();

    std::vector<rpc::RpcHandlePtr> hdls;
    for (size_t i = 0; i < eps.size(); ++i) {
      auto ep = eps[i];

      SaveOption option;
      option.set_ckp_type(CheckPointType(type_));
      option.set_ckp_target(CheckPointTarget(target_));
      option.set_version(version_);
      option.set_nfs_path(nfs_path_);
      option.set_queue(queue_);
      option.set_shard_idx(i);
      option.set_shard_num(eps.size());

      if (is_sparse) {
        option.set_variable_type(VariableType::VAR_EMBEDDING);
      } else {
        option.set_variable_type(VariableType::VAR_DENSE);
      }

      auto h = rpc_client_->SaveAsync(ep, varname_, option);
      hdls.push_back(h);
    }

    for (auto& h : hdls) {
      if (!h->Wait()) {
        LOG(WARNING) << "save var[" << varname_ << "] on ps[" << h->ep()
                     << "] fail. errmsg=" << h->errmsg();
        OP_REQUIRES(context, false, errors::InvalidArgument("save var timeout!"));
      } else {
        LOG(INFO) << "save var[" << varname_ << "] on ps[" << h->ep()
                  << "] success, ckp_type: " << type_
                  << ", target: " << target_
                  << ", nfs_path: " << nfs_path_;
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    monitor::RunStatus::Instance()->PushTime(
        monitor::kOpsSave, cost);

    LOG(INFO) << "save var[" << varname_ << "] target[" << target_
              << "] type[" << type_ << "] cost " << cost;
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::string varname_ = "";
  int32_t type_ = 0;
  int32_t target_ = 0;
  std::string nfs_path_ = "";
  std::string queue_ = "";

  int64_t version_;
};

REGISTER_KERNEL_BUILDER(Name("SniperSave").Device(DEVICE_CPU), SaveOp);

}  // namespace ops
}  // namespace sniper
