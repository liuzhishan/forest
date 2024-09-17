#include <memory>
#include <thread>

#include "glog/logging.h"
#include "trainer/core/operators/kernels/train_config.h"
#include "trainer/core/util/placement/auto_shard.h"
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
#include "trainer/core/base/semaphore.h"

// DEFINE_string(push_grad_varname_delimiter, "!@#", "");
DEFINE_bool(push_grad_bfloat16, false, "");

namespace sniper {
namespace ops {

using namespace tensorflow;

std::string JoinFloat(const float* arr, int n, const std::string& sep) {
  std::ostringstream oss;

  for (int i = 0; i < n; i++) {
    if (i < n - 1) {
      oss << arr[i] << sep;
    } else {
      oss << arr[i];
    }
  }

  return oss.str();
}

class PushGradOp : public OpKernel {
 public:
  explicit PushGradOp(OpKernelConstruction* context)
      : OpKernel(context), ctx_(context) {
    conf_file_ = "./train_config.json";
    OP_REQUIRES_OK(context, context->GetAttr("conf_file", &conf_file_));
    OP_REQUIRES_OK(context, context->GetAttr("trainer_id", &trainer_id_));
    train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);

    AutoShard::instance().add_placement(train_config_->placement());

    if (train_config_->debug_offline()) {
      push_thread_count_ = 1;
      LOG(INFO) << "debug offline, set push_thread_count_ = 1";
    }

    rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(
        trainer_id_);  // trainer_id
    for (int i = 0; i < push_thread_count_; ++i) {
      std::unique_ptr<std::thread> t(
          new std::thread(&PushGradOp::PushGradThread, this));
      push_threads_.push_back(std::move(t));
    }
    LOG(INFO) << "Trainer push gard op init. trainer_id: " << trainer_id_
              << ", sparse_fcs_size: " << train_config_->sparse_fcs().size();
  }

  ~PushGradOp() {
    exit_flag_.store(true);
    for (auto& t : push_threads_) {
      t->join();
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // const Tensor* batch_id;
    const Tensor* batch_id;
    const Tensor* var_list;
    const Tensor* gradient;
    const Tensor* eta;
    OP_REQUIRES_OK(ctx, ctx->input("batch_id", &batch_id));
    OP_REQUIRES_OK(ctx, ctx->input("var_list", &var_list));
    OP_REQUIRES_OK(ctx, ctx->input("gradient", &gradient));
    OP_REQUIRES_OK(ctx, ctx->input("eta", &eta));
    auto batch_id_vec = batch_id->vec<int64>();
    int64_t id = batch_id_vec(0);

    MsgPtr msg(new Msg());
    msg->batch_id = id;
    msg->eta = (eta->scalar<float>())();
    msg->var_list = *var_list;
    msg->gradient = *gradient;

    {
      if (train_config_->debug_offline()) {
        SemaphoreLoop::GetInstance().Acquire(2);
      }
      std::lock_guard<std::mutex> lk(grad_list_mutex_);
      grad_list_.push_back(msg);
      if (train_config_->debug_offline()) {
        SemaphoreLoop::GetInstance().Release(3);
      }
    }
    cond_.notify_one();
  }

 private:
  void PushGradThread() {
    auto batch_size = train_config_->batch_size();
    auto& emb_tables = train_config_->emb_tables();
    auto& input_sparse = train_config_->input_sparse();
    auto input_sparse_emb_table_names =
        train_config_->emb_table_names();

    if (train_config_->debug_offline()) {
      SemaphoreLoop::GetInstance().Acquire(3);
    }

    // 获取变量拓扑关系
    int32_t sum = 0;
    std::unordered_map<std::string, std::vector<std::string>> ps_var_names;
    std::unordered_map<std::string, std::vector<int32_t>> ps_var_size;
    std::unordered_map<std::string, std::vector<int32_t>> ps_var_idx;
    std::unordered_map<std::string, int32_t> ps_var_total_size;
    std::unordered_map<std::string, PushGradOption> ps_rpc_options;
    for (size_t i = 0; i < input_sparse.size(); ++i) {
      auto& emb_table = input_sparse_emb_table_names[i];

      auto it_table = emb_tables.find(emb_table);
      OP_REQUIRES(ctx_, it_table != emb_tables.end(),
                  errors::InvalidArgument("not found emb_table=" + emb_table));

      auto var_dim = it_table->second.dim();

      auto shard_eps = train_config_->placement()->GetEmbPlacement(emb_table);
      for (size_t j = 0; j < shard_eps.size(); ++j) {
        auto ep = shard_eps[j];

        ps_var_names[ep].push_back(emb_table);
        ps_var_size[ep].push_back(var_dim);
        ps_var_idx[ep].push_back(sum);
        if (ps_var_total_size.find(ep) == ps_var_total_size.end()) {
          ps_var_total_size[ep] = var_dim;
        } else {
          ps_var_total_size[ep] += var_dim;
        }

        ps_rpc_options[ep].set_batch_size(batch_size);
        ps_rpc_options[ep].add_field_idx(i);
        ps_rpc_options[ep].add_field_dim(var_dim);
      }
      sum += var_dim;
    }

    auto& auto_shard = AutoShard::instance();
    int update_time = 0;

    while (!exit_flag_) {
      std::unique_lock<std::mutex> lk(grad_list_mutex_);
      if (!cond_.wait_for(lk, std::chrono::seconds(1),
                          [this] { return !grad_list_.empty(); })) {
        // timeout
        continue;
      }
      auto msg = grad_list_.front();
      grad_list_.pop_front();
      lk.unlock();

      // 如果 shard 有更新，此处需要重新获取变量
      if (update_time != auto_shard.update_time()) {
        ps_var_names.clear();
        ps_var_size.clear();
        ps_var_idx.clear();
        ps_var_total_size.clear();
        ps_rpc_options.clear();

        for (size_t i = 0; i < input_sparse.size(); ++i) {
          auto& emb_table = input_sparse_emb_table_names[i];

          auto it_table = emb_tables.find(emb_table);
          OP_REQUIRES(
              ctx_, it_table != emb_tables.end(),
              errors::InvalidArgument("not found emb_table=" + emb_table));

          auto var_dim = it_table->second.dim();

          auto shard_eps = train_config_->placement()->GetEmbPlacement(emb_table);
          for (size_t j = 0; j < shard_eps.size(); ++j) {
            auto ep = shard_eps[j];

            ps_var_names[ep].push_back(emb_table);
            ps_var_size[ep].push_back(var_dim);
            ps_var_idx[ep].push_back(sum);
            if (ps_var_total_size.find(ep) == ps_var_total_size.end()) {
              ps_var_total_size[ep] = var_dim;
            } else {
              ps_var_total_size[ep] += var_dim;
            }

            ps_rpc_options[ep].set_batch_size(batch_size);
            ps_rpc_options[ep].add_field_idx(i);
            ps_rpc_options[ep].add_field_dim(var_dim);
          }
        }

        update_time = AutoShard::instance().update_time();
        LOG(INFO) << "update push grad ps vars, update_time: " << update_time;
      }

      auto batch_id = msg->batch_id;
      auto& var_list = msg->var_list;
      auto& grad = msg->gradient;
      auto eta = msg->eta;

      auto var_list_flat = var_list.flat<std::string>();

      auto start = std::chrono::steady_clock::now();
      auto grad_size = grad.dim_size(1);
      OP_REQUIRES(
          ctx_, sum == grad_size,
          errors::InvalidArgument("sum(var_grad_dim_size) != grad_size"));

      // ------------------- step1 ------------------
      // ps 纬度重新组包
      // why？ 如果按 var进行拆包，rpc调用会消耗大量 GPU机器的cpu资源，按
      // ps纬度组包，将资源消耗降低至1/10 ～ 1/5
      // 进而可以加上传输层压缩，可以进一步减少带宽占用
      auto grad_flat = grad.flat<float>();
      for (size_t i = 0; i < grad_flat.size(); i++) {
        OP_REQUIRES(ctx_, !std::isnan(grad_flat(i)),
                    errors::InvalidArgument(std::string("grad has nan! i: ") +
                                            std::to_string(i)));
      }

      std::unordered_map<std::string, Tensor> ps_grads;
      for (auto& it : ps_var_total_size) {
        auto ep = it.first;
        auto var_total_size = it.second;
        tensorflow::Tensor ps_grad = tensorflow::Tensor(
            tensorflow::DT_FLOAT, {batch_size, var_total_size});
        auto ps_grad_flat = ps_grad.flat<float>();

        auto var_names = ps_var_names[ep];
        auto var_sizes = ps_var_size[ep];
        auto var_idxs = ps_var_idx[ep];

        for (int i = 0; i < batch_size; i++) {
          int32_t ps_emb_size_idx = 0;
          for (size_t j = 0; j < var_names.size(); ++j) {
            auto emb_size = var_sizes[j];
            auto var_idx = var_idxs[j];
            std::copy_n(
                grad_flat.data() + i * grad_size + var_idx, emb_size,
                ps_grad_flat.data() + i * var_total_size + ps_emb_size_idx);
            ps_emb_size_idx += emb_size;

            // if (var_names[j] == "embedding_0") {
            //   LOG(INFO) << "before push grad, i: " << i
            //             << ", batch_id: " << batch_id
            //             << ", grad: " << JoinFloat(grad_flat.data() + i * grad_size + var_idx, emb_size, ",");
            // }
          }
          OP_REQUIRES(
              ctx_, ps_emb_size_idx == var_total_size,
              errors::InvalidArgument("ps_emb_size_idx != var_total_size"));
        }
        ps_grads[ep] = ps_grad;
      }

      // ------------------- step2 ------------------
      // 发送梯度更新包到各个 ps
      std::vector<rpc::RpcHandlePtr> hdls;
      for (auto& it : ps_grads) {
        auto ep = it.first;
        auto ps_grad = it.second;
        auto& varnames = ps_var_names[ep];
        auto& option = ps_rpc_options[ep];
        option.set_eta(eta);
        option.set_eps(train_config_->eps());
        option.set_l2(train_config_->l2());
        option.set_decay(train_config_->decay());
        option.set_version(train_config_->version());
        option.set_use_freq_scale(train_config_->use_freq_scale());

        if (FLAGS_push_grad_bfloat16) {
          auto ps_grad_flat = ps_grad.flat<float>();
          Tensor bfloat16_ps_grad(tensorflow::DT_BFLOAT16, ps_grad.shape());
          auto bfloat16_ps_grad_flat =
              bfloat16_ps_grad.flat<tensorflow::bfloat16>();
          tensorflow::FloatToBFloat16(ps_grad_flat.data(),
                                      bfloat16_ps_grad_flat.data(),
                                      ps_grad.NumElements());
          ps_grad = bfloat16_ps_grad;
        }
        // std::string ps_join_varname =
        //    absl::StrJoin(varnames, FLAGS_push_grad_varname_delimiter);
        std::string ps_join_varname = absl::StrJoin(varnames, ",");

        auto hdl = rpc_client_->PushGradAsync(ep, batch_id, ps_join_varname,
                                              option, ps_grad, Tensor());
        hdls.push_back(hdl);
      }
      // NOTE(dongxing) 为了性能，不做同步等待
      /*
      for (auto& h : hdls) {
        if (!h->Wait()) {
          LOG(WARNING) << "push grad at hub[" << h->ep()
                      << "] fail. error=" << h->errmsg();
          return;
        }
      }
      */

      if (train_config_->debug_offline()) {
        for (auto& h : hdls) {
          if (!h->Wait()) {
            LOG(WARNING) << "push grad at hub[" << h->ep()
                         << "] fail. error=" << h->errmsg();
            return;
          }
        }
        SemaphoreLoop::GetInstance().Release(4);
      }
      auto end = std::chrono::steady_clock::now();
      monitor::RunStatus::Instance()->PushTime(
          monitor::kOpsPushGrad,
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());
    }
    if (grad_list_.empty()) {
      LOG(INFO) << "Trainer PushGradThread exit, trainer_id: " << trainer_id_;
    } else {
      LOG(WARNING) << "Trainer PushGradThread exit, but queue is not empty, trainer_id: "
                   << trainer_id_
                   << ", grad_list.size(): " << grad_list_.size();
    }

  }

  OpKernelConstruction* ctx_;

  std::atomic<bool> exit_flag_{false};

  std::mutex grad_list_mutex_;
  std::condition_variable cond_;

  struct Msg {
    int64_t batch_id;
    Tensor var_list;
    Tensor gradient;
    float eta;
  };
  typedef std::shared_ptr<Msg> MsgPtr;
  std::deque<MsgPtr> grad_list_;

  std::vector<std::unique_ptr<std::thread>> push_threads_;

  std::string conf_file_;
  int32_t trainer_id_;
  TrainConfig* train_config_;

  tensorflow::int32 push_thread_count_ = 5;
  float eps_ = 1e-8;
  float decay_ = 0.0;
  float l2_ = 0.0;
  int version_ = 2;
  bool use_freq_scale_ = false;

  rpc::RPCClient* rpc_client_;
};

REGISTER_KERNEL_BUILDER(Name("PushGrad").Device(DEVICE_CPU), PushGradOp);

}  // namespace ops
}  // namespace sniper
