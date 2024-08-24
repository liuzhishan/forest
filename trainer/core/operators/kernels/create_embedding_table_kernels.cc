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
#include "core/util/placement/auto_shard.h"

namespace sniper {
namespace ops {

using namespace tensorflow;

class CreateEmbeddingTableOp : public OpKernel {
 public:
  explicit CreateEmbeddingTableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));

    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);  // trainer_id

    AutoShard::instance().add_placement(train_config_->placement());
  }

  ~CreateEmbeddingTableOp() {}

  void Compute(OpKernelContext* context) override {
    auto start = std::chrono::steady_clock::now();
    auto& emb_tables = train_config_->emb_tables();

    SPDLOG_INFO("start create table");

    std::vector<std::string> table_names;
    std::vector<rpc::RpcHandlePtr> hdls;
    for (auto& it : emb_tables) {
      std::string table_name = it.first;
      auto& table = it.second;

      SPDLOG_INFO("start create table, varname: {}", table_name);

      auto shard_eps = train_config_->placement()->GetEmbPlacement(table_name);

      CreateOption option;
      option.set_emb_size(table.dim());
      option.set_capacity(
          static_cast<uint32_t>(table.capacity() / shard_eps.size()));
      option.set_type(VAR_EMBEDDING);
      option.mutable_fields()->CopyFrom(table.fields());
      option.set_hit_limit(table.hit_limit());
      option.set_hash_size(table.hash_bucket_size());
      option.set_delete_var(false);

      option.set_use_param_vector(train_config_->use_param_vector());

      //TODO for test default opt is adam
      option.set_optimizer(train_config_->optimizer());
      option.set_beta1(train_config_->beta1());
      option.set_beta2(train_config_->beta2());
      option.set_embedding_lr(train_config_->embedding_lr());
      option.set_embedding_eps(train_config_->embedding_eps());
      option.set_feature_inclusion_freq(train_config_->feature_inclusion_freq());

      for (size_t i = 0; i < shard_eps.size(); ++i) {
        auto ep = shard_eps[i];
        option.set_shard_idx(i);
        option.set_shard_num(shard_eps.size());

        auto h = rpc_client_->CreateAsync(ep, table_name, option, 180000);
        hdls.push_back(h);
        table_names.push_back(table_name);
      }
    }

    for (size_t i = 0; i < hdls.size(); ++i) {
      auto& h = hdls[i];
      auto& table_name = table_names[i];
      if (!h->Wait()) {
        LOG(WARNING) << "create embedding_table[" << table_name << "] on ps["
                     << h->ep() << "] fail. errmsg=" << h->errmsg();
        return;
      } else {
        LOG(INFO) << "create embedding_table[" << table_name << "] on ps["
                  << h->ep() << "] success.";
      }
    }

    SPDLOG_INFO("create table done");
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
};  // namespace ops

REGISTER_KERNEL_BUILDER(Name("CreateEmbeddingTable").Device(DEVICE_CPU),
                        CreateEmbeddingTableOp);

}  // namespace ops
}  // namespace sniper
