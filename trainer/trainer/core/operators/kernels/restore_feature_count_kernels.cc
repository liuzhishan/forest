#include "glog/logging.h"
#include "absl/strings/str_split.h"
#include "absl/strings/str_join.h"
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

class RestoreFeatureCountOp : public OpKernel {
 public:
  explicit RestoreFeatureCountOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);

    AutoShard::instance().add_placement(train_config_->placement());

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("paths", &paths_));
  }
  ~RestoreFeatureCountOp() {}

  void Compute(OpKernelContext* context) override {
    auto start = std::chrono::steady_clock::now();

    if (varname_.rfind("embedding_", 0) != 0) {
      return;
    }

    std::vector<std::string> eps = train_config_->placement()->GetEmbPlacement(varname_);
    int32_t total_iteration = 60;
    bool restore_finish = false;
    bool restore_error = false;
    size_t success_cnt = 0;

    LOG(INFO) << "start restore_feature_count, varname: " << varname_
              << ", paths: " << absl::StrJoin(paths_, ",");

    while (!restore_finish && !restore_error && (--total_iteration > 0)) {
      for (size_t idx = 0; idx < eps.size(); idx++) {
        auto ep = eps[idx];
        RestoreFeatureCountOption option;
        option.set_varname(varname_);
        option.set_shard_idx(idx);
        option.set_shard_num(eps.size());
        for (auto& p: absl::StrSplit(paths_[idx], ",")) {
          option.add_paths(std::string(p));
        }
        rpc::TensorResponse response;
        auto h = rpc_client_->RestoreFeatureCountAsync(ep, varname_, option, &response);
        if (!h->Wait()) {
          LOG(INFO) << "failed" << ", varname: " << varname_;
          LOG(WARNING) << "restore feature count var[" << varname_ << "] on ps[" << h->ep()
                      << "] fail. errmsg=" << h->errmsg();
        } else {
          RestoreFeatureCountOption ret_option;
          response.meta().options().UnpackTo(&ret_option);
          if (!ret_option.errmsg().empty()) {
            LOG(WARNING) << "Trainer restore feature count fail, trainer_id: " << trainer_id_
                         << ", ep: " << ep
                         << ", errmsg: " << ret_option.errmsg();
            restore_error = true;
          } else {
            if (ret_option.finish()) {
              success_cnt += 1;
            }
          }

          OP_REQUIRES(context,
                      restore_error == false,
                      errors::InvalidArgument(std::string("restore feature count fail: ") + ret_option.errmsg()));
        }

        if (success_cnt == eps.size()) {
          LOG(INFO) << "one ep succcess, "
                    << ", ep: " << ep
                    << ", idx: " << idx
                    << ", varname: " << varname_
                    << ", eps.size(): " << eps.size()
                    << ", success_cnt: " << success_cnt;
          restore_finish = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(20));
      }
    }

    OP_REQUIRES(context, restore_finish == true,
                errors::InvalidArgument("restore feature count timeout, ", "var: " + varname_ ));

    auto end = std::chrono::steady_clock::now();
    monitor::RunPushTime(monitor::kOpsRestoreFeatureCount, end - start);

    LOG(INFO) << "restore feature count kernel success, var: " << varname_
              << ", paths: " << absl::StrJoin(paths_, ",")
              << ", cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::string varname_ = "";
  std::vector<std::string> paths_;
};

REGISTER_KERNEL_BUILDER(Name("RestoreFeatureCount").Device(DEVICE_CPU), RestoreFeatureCountOp);

}  // namespace ops
}  // namespace sniper
