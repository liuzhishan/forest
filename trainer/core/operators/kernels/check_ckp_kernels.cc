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

class CheckCkpOp : public OpKernel {
 public:
  explicit CheckCkpOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id

    OP_REQUIRES_OK(context, context->GetAttr("ckp_type", &ckp_type_));
    OP_REQUIRES_OK(context, context->GetAttr("ckp_target", &ckp_target_));
  }
  ~CheckCkpOp() {}

  void Compute(OpKernelContext* context) override {
    const Tensor& version = context->input(0);
    int64_t ckp_version = (version.scalar<int64_t>())();

    auto& ps_eps = train_config_->ps_eps();

    OP_REQUIRES(context, ps_eps.size() > 0,
                errors::InvalidArgument("ps eps empty"));

    HeartbeatOption option;
    option.set_st_type(HeartbeatOption::TYPE_CHECK);
    option.mutable_check_st()->set_version(ckp_version);
    option.mutable_check_st()->set_ckp_type((klearn::CheckPointType)ckp_type_);
    option.mutable_check_st()->set_ckp_target((klearn::CheckPointTarget)ckp_target_);

    std::vector<rpc::RpcHandlePtr> hdls;
    auto& scheduler_ep = ps_eps[0];

    LOG(INFO) << "CheckCkp started:" << option.DebugString();
    auto total_iteration = 180;
    bool finished = false;
    while (--total_iteration > 0) {
      rpc::TensorResponse response;
      auto h = rpc_client_->HeartbeatAsync(scheduler_ep, option, &response);
      if (!h->Wait()) {
        LOG(WARNING) << "CheckCkp version[" << ckp_version << "] on ps[" << h->ep()
                     << "] fail. errmsg=" << h->errmsg();
        OP_REQUIRES(context, false, errors::InvalidArgument("CheckCkp timeout!"));
      } else {
        HeartbeatOption ret_option;
        response.meta().options().UnpackTo(&ret_option);
        if (!ret_option.check_st().errmsg().empty()) {
          SPDLOG_WARN("Trainer{0} CheckCkp fail. ep={1}, errmsg={2}",
                      trainer_id_, h->ep(), ret_option.check_st().errmsg());
          OP_REQUIRES(context, false, errors::InvalidArgument("CheckCkp error!"));
          break;
        } else {
          if (!ret_option.check_st().finish()) {
            LOG(INFO) << "wait for CheckCkp:" << option.DebugString();
          } else {
            finished = true;
            if (ret_option.check_st().succ()) {
              LOG(INFO) << "CheckCkp found succed: " << option.DebugString();
            } else {
              LOG(WARNING) << "CheckCkp found failed: " << option.DebugString();
              OP_REQUIRES(context, false, errors::InvalidArgument("CheckCkp failed!"));
            }
            break;
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(60));
    }
    if (!finished && total_iteration <= 0) {
      LOG(WARNING) << "CheckCkp timeout";
      OP_REQUIRES(context, false, errors::InvalidArgument("CheckCkp timeout!"));
    }
  }

 private:
  int32_t ckp_type_;
  int32_t ckp_target_;
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;
};  // namespace ops

REGISTER_KERNEL_BUILDER(Name("CheckCkp").Device(DEVICE_CPU), CheckCkpOp);

}  // namespace ops
}  // namespace sniper
