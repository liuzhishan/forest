#include <memory>
#include <thread>

#include "glog/logging.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_client.h"
#include "trainer/core/util/monitor/run_status.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "trainer/core/util/placement/auto_shard.h"
#include "trainer/core/operators/kernels/train_config.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class PullVariableOp : public OpKernel {
 public:
  explicit PullVariableOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id

    AutoShard::instance().add_placement(train_config_->placement());

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("vartype", &vartype_));
  }
  ~PullVariableOp() {}

  void Compute(OpKernelContext* ctx) override {
    Tensor var = ctx->mutable_input(0, true);
    Tensor var_opt = ctx->mutable_input(1, true);

    if ((VariableType)vartype_ == VAR_DENSE) {
      auto ep = train_config_->placement()->GetDensePlacement(varname_);
      PullOption option;
      option.set_variable_type(VariableType::VAR_DENSE);
      rpc::TensorResponse response;
      auto h = rpc_client_->PullAsync(ep, 0, varname_, option, Tensor(),
                                      Tensor(), &response);
      if (!h->Wait()) {
        OP_REQUIRES(ctx, false,
                    errors::Internal("pull variable[" + varname_ +
                                     "] at ps[" + h->ep() +
                                     "] fail, errmsg=" + h->errmsg()));
      }
      auto ret = response.tensor1();
      if (ret.shape() == var.shape() && ret.dtype() == var.dtype()) {
        auto ret_flat = ret.flat<float>();
        auto var_flat = var.flat<float>();
        std::copy_n(ret_flat.data(), ret_flat.size(), var_flat.data());
      } else {
        std::string msg;

        msg.append("pull variale failed, shape or dtype not match, varname: ")
          .append(varname_)
          .append(", ret.shape: ")
          .append(absl::StrJoin(ret.shape().dim_sizes(), ","))
          .append(", ret.dtype: ")
          .append(std::to_string(ret.dtype()))
          .append(", var.shape: ")
          .append(absl::StrJoin(var.shape().dim_sizes(), ","))
          .append(", ret.dtype: ")
          .append(std::to_string(var.dtype()));

        OP_REQUIRES(ctx, false, errors::Internal(msg));
      }
      LOG(INFO) << "pull variable success, varname: " << varname_;
    } else if ((VariableType)vartype_ == VAR_EMBEDDING) {
      auto shard_eps = train_config_->placement()->GetEmbPlacement(varname_);
      auto shard_num = shard_eps.size();

      auto var_flat = var.flat<float>();
      auto var_opt_flat = var_opt.flat<float>();

      auto capacity = var.dim_size(0);
      auto emb_size = var.dim_size(1);

      int64_t total_key_count = 0;
      for (size_t shard = 0; shard < shard_num; ++shard) {
        int64_t cur_progress = 0;
        bool is_completed = false;
        auto ep = shard_eps[shard];
        while (!is_completed) {
          rpc::TensorResponse response;

          PullOption option;

          option.set_progress(cur_progress);
          option.set_variable_type(VariableType::VAR_EMBEDDING);

          auto h = rpc_client_->PullAsync(ep, 0, varname_, option,
                                          tensorflow::Tensor(),
                                          tensorflow::Tensor(), &response);
          if (!h->Wait()) {
            OP_REQUIRES(ctx, false,
                        errors::Internal("pull variable[" + varname_ +
                                         "] at ps[" + h->ep() +
                                         "] fail, errmsg=" + h->errmsg()));
          }
          auto key_tensor = response.tensor1();
          auto val_tensor = response.tensor2();
          auto key_flat = key_tensor.flat<tensorflow::uint64>();
          auto val_flat = val_tensor.flat<float>();
          for (auto i = 0; i < key_flat.size(); ++i) {
            auto key = key_flat(i);
            if (key >= capacity) {
              LOG(WARNING) << "pull variable fail, varname: " << varname_
                           << ", key: " << key
                           << ", capacity: " << capacity;
            }
            std::copy_n(val_flat.data() + i * emb_size * 2, emb_size,
                        var_flat.data() + key * emb_size);
            std::copy_n(val_flat.data() + i * emb_size * 2 + emb_size, emb_size,
                        var_opt_flat.data() + key * emb_size);
            total_key_count++;
          }
          PullOption ret_option;
          response.meta().options().UnpackTo(&ret_option);
          is_completed = ret_option.completed();
          cur_progress = ret_option.progress();
        }
      }
      LOG(INFO) << "pull variable success, varname: " << varname_
                << ", total_key_count: " << total_key_count;
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("not valid vartype: " +
                                          std::to_string(vartype_)));
    }

    // 此 op 没有 output, outputs_.size() 是 0，以下两个函数会 core，因此注释掉。
    // core 的位置是 op_kernel.cc 的 set_output_ref 函数中，如下位置:
    // CHECK_LT(index, outputs_.size());
    //
    // 但是不知道为什么 tf1.14 没有报错。看 1.14 的代码也是会 core 的。
    //
    // ctx->forward_ref_input_to_ref_output(0, 0);
    // ctx->forward_ref_input_to_ref_output(1, 1);
  }

 private:
  std::string conf_file_;
  int32_t trainer_id_;
  TrainConfig* train_config_;

  std::string varname_;
  int32_t vartype_;

  rpc::RPCClient* rpc_client_;
};  // namespace ops

REGISTER_KERNEL_BUILDER(Name("PullVariable").Device(DEVICE_CPU),
                        PullVariableOp);

}  // namespace ops
}  // namespace sniper
