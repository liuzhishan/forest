#include <memory>
#include <thread>

#include "glog/logging.h"

#include "absl/strings/str_join.h"
#include "trainer/core/operators/kernels/train_config.h"
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

DEFINE_int32(max_push_key_count, 100000, "");

namespace sniper {
namespace ops {

using namespace tensorflow;

class PushVariableOp : public OpKernel {
 public:
  explicit PushVariableOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));

    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);

    OP_REQUIRES_OK(context, context->GetAttr("varname", &varname_));
    OP_REQUIRES_OK(context, context->GetAttr("vartype", &vartype_));
  }
  ~PushVariableOp() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& var = ctx->input(0);
    const Tensor& var_opt = ctx->input(1);

    if ((VariableType)vartype_ == VAR_DENSE) {
      auto ep = train_config_->placement()->GetDensePlacement(varname_);
      PushOption option;
      option.set_variable_type(VariableType::VAR_DENSE);

      LOG(INFO) << "push dense variable, varname: " << varname_
                << ", ep: " << ep
                << ", tensor size: { "
                << absl::StrJoin(var.shape().dim_sizes(), ", ")
                << " }";

      auto h = rpc_client_->PushAsync(ep, 0, varname_, option, var, Tensor());
      if (!h->Wait()) {
        OP_REQUIRES(ctx, false,
                    errors::Internal("push variable[" + varname_ +
                                     "] at ps[" + h->ep() +
                                     "] fail, errmsg=" + h->errmsg()));
      }
      LOG(INFO) << "push dense variable " << varname_ << " at ps " << h->ep() << " success";
    } else if ((VariableType)vartype_ == VAR_EMBEDDING) {
      auto shard_eps = train_config_->placement()->GetEmbPlacement(varname_);
      auto shard_num = shard_eps.size();
      std::vector<rpc::RpcHandlePtr> hdls;

      auto var_flat = var.flat<float>();
      auto var_opt_flat = var_opt.flat<float>();

      auto capacity = var.dim_size(0);
      auto emb_size = var.dim_size(1);
      for (int i = 0; i < capacity; i += FLAGS_max_push_key_count) {
        int n = std::min<int>(FLAGS_max_push_key_count, capacity - i);
        tensorflow::Tensor key_tensor =
            tensorflow::Tensor(tensorflow::DT_UINT64, {n});
        auto key_flat = key_tensor.flat<tensorflow::uint64>();

        tensorflow::Tensor val_tensor =
            tensorflow::Tensor(tensorflow::DT_FLOAT, {n, 2 * emb_size});
        auto val_flat = val_tensor.flat<float>();

        for (int j = 0; j < n; ++j) {
          int pos = i + j;
          key_flat(j) = pos;
          std::copy_n(var_flat.data() + pos * emb_size, emb_size,
                      val_flat.data() + j * emb_size * 2);
          std::copy_n(var_opt_flat.data() + pos * emb_size, emb_size,
                      val_flat.data() + j * emb_size * 2 + emb_size);
        }

        std::vector<rpc::RpcHandlePtr> hdls;

        for (size_t shard = 0; shard < shard_num; ++shard) {
          auto ep = shard_eps[shard];
          PushOption option;
          option.set_variable_type(VariableType::VAR_EMBEDDING);

          LOG(INFO) << "push embedding variable, varname: " << varname_
                    << ", shard: " << shard
                    << ", shard_num: " << shard_num
                    << ", ep: " << ep
                    << ", key_tensor size: " << n
                    << ", value tensor size: { " << n << ", " << 2 * emb_size << " }";

          auto h = rpc_client_->PushAsync(ep, 0, varname_, option, key_tensor,
                                          val_tensor);

          hdls.push_back(h);
        }

        for (size_t i = 0; i < hdls.size(); ++i) {
          auto& h = hdls[i];
          if (!h->Wait()) {
            OP_REQUIRES(ctx, false,
                        errors::Internal("push variable[" + varname_ +
                                         "] at ps[" + h->ep() +
                                         "] fail, errmsg=" + h->errmsg()));
          } else {
            LOG(INFO) << "push embedding variable " << varname_
                      << " at ps " << h->ep()
                      << " success";
          }
        }
      }
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("not valid vartype: " +
                                          std::to_string(vartype_)));
    }
  }

 private:
  std::string conf_file_;
  int32_t trainer_id_;
  TrainConfig* train_config_;

  std::string varname_;
  int32_t vartype_;

  rpc::RPCClient* rpc_client_;
};

REGISTER_KERNEL_BUILDER(Name("PushVariable").Device(DEVICE_CPU),
                        PushVariableOp);

}  // namespace ops
}  // namespace sniper
