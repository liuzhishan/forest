#include <sstream>
#include <mutex>

#include "include/json/json.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_client.h"
#include "trainer/core/util/monitor/run_status.h"
#include "trainer/core/operators/kernels/train_config.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class CountFeatureOp : public OpKernel {
 public:
  explicit CountFeatureOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";

    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));

    train_config_ = TrainConfig::GetInstance(conf_file_, 0);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(0);

    hub_eps_ = train_config_->hub_eps();
    hub_over_.assign(hub_eps_.size(), false);
  }

  ~CountFeatureOp() {}

  inline bool is_over() {
    for (auto b : hub_over_) {
      if (!b) {
        return false;
      }
    }
    return true;
  }

  inline int32_t get_next_hub() {
    std::unique_lock<std::mutex> lk(mu_);
    for (size_t i = 0; i < hub_over_.size(); ++i) {
      int can_use = (hub_idx_ + i) % hub_over_.size();
      if (!hub_over_[can_use]) {
        ++hub_idx_;
        hub_idx_ %= hub_over_.size();
        return can_use;
      }
    }
    LOG(INFO) << "get next hub fail";
    return -1;
  }

  inline void set_hub_over(int32_t idx) {
    std::unique_lock<std::mutex> lk(mu_);
    LOG(INFO) << "hub " << idx << " is over!!!";
    hub_over_[idx] = true;
  }

  void Compute(OpKernelContext* context) override {
    auto& ps_eps = train_config_->ps_eps();

    OP_REQUIRES(context, ps_eps.size() > 0, errors::InvalidArgument("ps eps empty"));

    int64_t cnt = 0;
    while (true) {
      auto idx = get_next_hub();
      if (idx == -1) {
        continue;
      }

      auto start = std::chrono::steady_clock::now();
      rpc::TensorResponse response;
      auto hdl = rpc_client_->ReadSampleAsync(hub_eps_[idx], &response);
      if (!hdl->Wait()) {
        LOG(INFO) << "ReadSample fail, sleep 1000 microseconds";
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        continue;
      }

      ++cnt;
      if (cnt % 1000 == 0) {
        LOG(INFO) << "not finish, read " << cnt << " batch";
      }

      ReadSampleOption option;
      response.meta().options().UnpackTo(&option);
      if (option.over()) {
        set_hub_over(idx);
        if (is_over()) {
          break;
        } else {
          continue;
        }
      }

      if (option.need_wait()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        continue;
      }

      auto end = std::chrono::steady_clock::now();
      monitor::RunPushTime(monitor::kOpsReadSample, end - start);
    }

    LOG(INFO) << "raad sample done, sleep 5 seconds";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    Json::Value counter;
    CountFeatureOption option;
    for (auto& ps: train_config_->ps_eps()) {
      rpc::TensorResponse response;
      auto h = rpc_client_->CountFeatureAsync(ps, option, &response);
      if (h->Wait()) {
        CountFeatureOption count_result;
        response.meta().options().UnpackTo(&count_result);
        for (auto it = count_result.count_feature().begin(); it != count_result.count_feature().end(); it++) {
          counter[it->first] = counter[it->first].asInt64() + it->second;
        }
      } else {
        LOG(WARNING) << "CountFeature fail, ps: " << ps << ", errmsg=" << h->errmsg();
      }
    }

    std::stringstream ss;
    ss << counter;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    output_tensor->flat<string>()(0) = ss.str();
  }

 private:
  std::string conf_file_;

  TrainConfig* train_config_;

  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;

  std::vector<std::string> hub_eps_;
  mutable std::mutex mu_;

  // hub server queue is over?
  std::vector<bool> hub_over_;

  int32_t hub_idx_ = 0;
};  // namespace ops

REGISTER_KERNEL_BUILDER(Name("CountFeature").Device(DEVICE_CPU), CountFeatureOp);

}  // namespace ops
}  // namespace sniper
