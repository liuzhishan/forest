#include "core/operators/kernels/train_config.h"
#include "core/proto/meta.pb.h"
#include "core/rpc/grpc/grpc_client.h"
#include "core/util/monitor/run_status.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class CreateDenseOp : public OpKernel {
 public:
  explicit CreateDenseOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id
    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
  }
  ~CreateDenseOp() {}

  void Compute(OpKernelContext* context) override {
    const Tensor& var = context->input(0);

    auto start = std::chrono::steady_clock::now();
    CreateOption option;
    option.set_type(VAR_DENSE);

    auto dims = var.shape().dim_sizes();
    for (auto& dim : dims) {
      option.mutable_shape()->add_dim(dim);
    }

    auto ep = train_config_->placement()->GetDensePlacement(varname_);
    auto h = rpc_client_->CreateAsync(ep, varname_, option);

    if (!h->Wait()) {
      LOG(WARNING) << "create var[" << varname_ << "] on ps[" << h->ep()
                   << "] fail. errmsg=" << h->errmsg();
    } else {
      LOG(INFO) << "create var[" << varname_ << "] on ps[" << h->ep()
                << "] success.";
    }

    auto end = std::chrono::steady_clock::now();
    monitor::RunStatus::Instance()->PushTime(
        monitor::kOpsCreate,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::string varname_;
};

REGISTER_KERNEL_BUILDER(Name("CreateDense").Device(DEVICE_CPU), CreateDenseOp);

}  // namespace ops
}  // namespace sniper
