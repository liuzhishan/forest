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

namespace sniper {
namespace ops {

using namespace tensorflow;

class FreezeOp : public OpKernel {
 public:
  explicit FreezeOp(OpKernelConstruction* context) : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id

    OP_REQUIRES_OK(context, context->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(context, context->GetAttr("dense_vars", &dense_vars_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dense_var_queues", &dense_var_queues_));
    OP_REQUIRES_OK(context, context->GetAttr("sparse_vars", &sparse_vars_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sparse_var_queues", &sparse_var_queues_));
  }
  ~FreezeOp() {}

  void Compute(OpKernelContext* context) override {
    auto& hub_eps = train_config_->hub_eps();
    auto& ps_eps = train_config_->ps_eps();
    auto input_sparse_emb_table_names =
        train_config_->emb_table_names();

    OP_REQUIRES(context, hub_eps.size() > 0,
                errors::InvalidArgument("hub eps empty"));
    OP_REQUIRES(context, ps_eps.size() > 0,
                errors::InvalidArgument("ps eps empty"));
    int emb_table_nums = 0;
    std::map<std::string, bool> name_map;
    for (auto& emb_table_name : input_sparse_emb_table_names) {
      if (name_map.find(emb_table_name) != name_map.end()) {
        continue;
      }
      emb_table_nums++;
      name_map[emb_table_name] = true;
    }
    LOG(INFO) << "sparse vars size: " << sparse_vars_.size()
              << ", emb_table_nums: " << emb_table_nums;
    OP_REQUIRES(context,
                emb_table_nums == sparse_vars_.size(),
                errors::InvalidArgument("sparse vars size not match"));

    FreezeOption option;
    if (train_config_->embedding_table_store_type().size() == 0) {
      option.set_embedding_table_store_type("cuckoohash_map");
    } else {
      option.set_embedding_table_store_type(train_config_->embedding_table_store_type());
    }
    auto btq_push_limit_total = train_config_->btq_push_limit_total();
    if (btq_push_limit_total == -1) {
      option.set_btq_push_limit(-1);
    } else if (btq_push_limit_total > 0) {
      option.set_btq_push_limit(btq_push_limit_total / ps_eps.size());
    }
    option.set_ckp_save_btq_incr_sparse_step(train_config_->ckp_save_btq_incr_sparse_step());
    option.set_model_name(model_name_);
    option.set_scheduler_ep(ps_eps[0]);
    *(option.mutable_hub_eps()) = {hub_eps.begin(), hub_eps.end()};
    *(option.mutable_ps_eps()) = {ps_eps.begin(), ps_eps.end()};
    *(option.mutable_dense_vars()) = {dense_vars_.begin(), dense_vars_.end()};
    *(option.mutable_dense_var_queues()) = {dense_var_queues_.begin(),
                                            dense_var_queues_.end()};
    *(option.mutable_sparse_vars()) = {sparse_vars_.begin(),
                                       sparse_vars_.end()};
    *(option.mutable_sparse_var_queues()) = {sparse_var_queues_.begin(),
                                             sparse_var_queues_.end()};
    for (auto& it: train_config_->ps_shard()) {
      RepeatedString arr;
      for (auto& ps_name: it.second) {
        arr.mutable_value()->Add()->assign(ps_name);
      }
      (*(option.mutable_ps_shard()))[it.first] = arr;
    }

    std::vector<rpc::RpcHandlePtr> hdls;
    for (size_t i = 0; i < ps_eps.size(); ++i) {
      auto& ep = ps_eps[i];
      if (i == 0) {
        option.set_is_scheduler(true);
      } else {
        option.set_is_scheduler(false);
      }
      auto h = rpc_client_->FreezeAsync(ep, option);
      hdls.push_back(h);
    }

    for (auto& h : hdls) {
      if (!h->Wait()) {
        LOG(WARNING) << "freeze on ps[" << h->ep()
                     << "] fail. errmsg=" << h->errmsg();
        break;
      } else {
        LOG(INFO) << "freeze on ps[" << h->ep() << "] success.";
      }
    }
  }

 private:
  std::string conf_file_;
  TrainConfig* train_config_;
  int32_t trainer_id_;
  rpc::RPCClient* rpc_client_;
  int32_t total_sparse_shard_;

  std::string model_name_;
  std::vector<std::string> dense_vars_;
  std::vector<std::string> dense_var_queues_;
  std::vector<std::string> sparse_vars_;
  std::vector<std::string> sparse_var_queues_;
};  // namespace ops

REGISTER_KERNEL_BUILDER(Name("Freeze").Device(DEVICE_CPU), FreezeOp);

}  // namespace ops
}  // namespace sniper
