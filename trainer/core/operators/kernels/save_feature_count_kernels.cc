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

class SaveFeatureCountOp : public OpKernel {
 public:
  explicit SaveFeatureCountOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("nfs_path", &nfs_path_));
  }
  ~SaveFeatureCountOp() {}

  void Compute(OpKernelContext* context) override {
    auto start = std::chrono::steady_clock::now();

    if (varname_.rfind("embedding_", 0) != 0) {
      return;
    }

    std::vector<std::string> eps = train_config_->placement()->GetEmbPlacement(varname_);
    int32_t total_iteration = 60;
    bool save_finish = false;
    bool save_error = false;
    size_t success_cnt = 0;

    while (!save_finish && !save_error && (--total_iteration > 0)) {
      for (size_t idx = 0; idx < eps.size(); idx++) {
        auto ep = eps[idx];
        SaveFeatureCountOption option;
        option.set_varname(varname_);
        option.set_shard_idx(idx);
        option.set_shard_num(eps.size());
        option.set_nfs_path(nfs_path_);
        rpc::TensorResponse response;
        auto h = rpc_client_->SaveFeatureCountAsync(ep, varname_, option, &response);
        if (!h->Wait()) {
          LOG(INFO) << "failed" << ", varname: " << varname_;
          LOG(WARNING) << "save feature count var[" << varname_ << "] on ps[" << h->ep()
                      << "] fail. errmsg=" << h->errmsg();
        } else {
          SaveFeatureCountOption ret_option;
          response.meta().options().UnpackTo(&ret_option);
          if (!ret_option.errmsg().empty()) {
            SPDLOG_WARN("Trainer{0} save feature count fail. ep={1}, errmsg={2}",
                        trainer_id_, ep, option.errmsg());
            save_error = true;
          } else {
            if (ret_option.finish()) {
              success_cnt += 1;
            }
          }

          OP_REQUIRES(context,
                      save_error == false,
                      errors::InvalidArgument(std::string("save feature count fail: ") + ret_option.errmsg()));
        }

        LOG(INFO) << "one ep succcess"
                  << ", ep: " << ep
                  << ", idx: " << idx
                  << ", varname: " << varname_
                  << ", eps.size(): " << eps.size()
                  << ", success_cnt: " << success_cnt;

        if (success_cnt == eps.size()) {
          save_finish = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(20));
      }
    }

    OP_REQUIRES(context, save_finish == true,
                errors::InvalidArgument("save feature count timeout, ", "var: " + varname_ ));

    auto end = std::chrono::steady_clock::now();
    monitor::RunPushTime(monitor::kOpsSaveFeatureCount, end - start);

    LOG(INFO) << "save feature count kernel success, var: " << varname_
              << ", cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::string varname_ = "";
  std::string nfs_path_ = "";
};

REGISTER_KERNEL_BUILDER(Name("SaveFeatureCount").Device(DEVICE_CPU), SaveFeatureCountOp);

}  // namespace ops
}  // namespace sniper
