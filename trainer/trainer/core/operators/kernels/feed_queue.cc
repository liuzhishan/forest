#include "trainer/core/operators/kernels/feed_queue.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "absl/strings/str_join.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "trainer/core/base/semaphore.h"
#include "trainer/core/base/util.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/rpc/grpc/grpc_client.h"
#include "trainer/core/util/monitor/run_status.h"

// DEFINE_string(varname_delimiter, "!@#", "");

namespace sniper {
namespace ops {

FeedQueue::FeedQueue(const std::string& conf_file, int32_t trainer_id, int32_t work_mode)
    : conf_file_(conf_file), trainer_id_(trainer_id), work_mode_(work_mode), ready_to_consume_(false) {
  train_config_ = TrainConfig::GetInstance(conf_file_, trainer_id_);
  hub_eps_ = train_config_->hub_eps();
  hub_over_.assign(hub_eps_.size(), false);

  parallel_ = train_config_->feed_worker();

  if (train_config_->debug_offline()) {
    parallel_ = 1;
    LOG(INFO) << "debug offline, set parallel = 1";
  }

  use_auto_shard_ = train_config_->use_auto_shard();
  AutoShard::instance().init(train_config_->ps_eps().size(), train_config_->ps_shard_by_field(),
                             train_config_->top_ps(), train_config_->top_field(),
                             train_config_->field_shard_limit(), train_config_->update_shard_limit(),
                             train_config_->step_limit(), train_config_->is_move_shard());

  init();
  LOG(INFO) << "Trainer feed queue created. trainer_id: " << trainer_id_
            << ", hub_eps_size: " << hub_eps_.size()
            << ", parallel: " << parallel_
            << ", queue_size: " << queue_size_
            << ", work_mode: " << work_mode_;
}

void FeedQueue::init() {
  rpc_client_ = rpc::RPCClient::GetInstance<rpc::GRPCClient>(trainer_id_);  // trainer_id

  for (int i = 0; i < parallel_; ++i) {
    threads_.push_back(new std::thread(&FeedQueue::consume, this, i));
  }

  stat_bg_ = new std::thread([&]() {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        if (finished_) {
          break;
        }
        if (queue_.size() < 5) {
          LOG(WARNING) << "Trainer feed queue size({1}) is not rich. trainer_id: " << trainer_id_
                       << ", queue.size(): " << queue_.size()
                       << ", training will bound at prefetch, "
                       << " you can check trainer network and hub / ps resource.";

        } else {
          LOG(INFO) << "trainer_id: " << trainer_id_
                    << ", feed queue size: " << queue_.size();
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });
}

std::string join_float(float* p, int32_t n) {
  std::string s;
  for (int i = 0; i < n; i++) {
    s += std::to_string(*(p + i)) + ",";
  }

  s += "|";
  return s;
}

bool FeedQueue::feed(FeatureColumn*& feature, bool* over) {
  ready_to_consume_ = true;
  std::unique_lock<std::mutex> lock(mu_);
  while (queue_.empty() && !finished_ && !is_over()) {
    cv_.wait_for(lock, std::chrono::microseconds(100));
  }
  if (finished_) return false;
  if (!queue_.empty()) {
    feature = queue_.front();
    queue_.pop();
    lock.unlock();
    cv_.notify_one();  // 通知队列已经不再满了

    if (feature != nullptr) {
      return true;
    }
    LOG(ERROR) << "Trainer feature is null, trainer_id: " << trainer_id_;
    return false;
  } else if (is_over()) {
    *over = true;
    return true;
  } else {
    return false;
  }
}

void FeedQueue::reset() {
  std::unique_lock<std::mutex> lock(mu_);
  hub_over_.assign(hub_eps_.size(), false);
  finished_ = false;
}

void FeedQueue::consume(int thread_id) {
  auto& input_sparse = train_config_->input_sparse();
  auto& input_sparse_dim = train_config_->input_sparse_dim();
  auto& input_sparse_emb_table_names = train_config_->emb_table_names();
  auto batch_size = train_config_->batch_size();
  auto label_size = train_config_->label_size();
  auto dense_total_size = train_config_->dense_total_size();
  const auto& ps_to_index = train_config_->ps_to_index();

  bool last_success = false;
  bool is_normal = false;

  auto& auto_shard = AutoShard::instance();

  auto feature = NewFeatureColumn();
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (finished_) {
        break;
      }
      if (is_over()) {
        LOG(INFO) << "Trainer worker is over, trainer_id: " << trainer_id_;
        break;
      }
    }

    auto idx = get_next_hub();
    if (idx == -1) {
      continue;
    }

    if (train_config_->debug_offline()) {
      if (!SemaphoreLoop::GetInstance().IsStart() && last_success) {
        int acquire_num = 4;
        // for predict;
        if (work_mode_ == 2) {
          acquire_num = 2;
        }
        SemaphoreLoop::GetInstance().Acquire(acquire_num);
      }
      SemaphoreLoop::GetInstance().SetIsStart(false);
      last_success = false;
    }

    bool is_in_update_shard = false;
    if (use_auto_shard_) {
      // 更新 shard
      if (!auto_shard.is_finish() && auto_shard.is_ready()) {
        if (thread_id == 0) {
          std::unique_lock<std::mutex> lk_shard(mu_shard_);

          is_in_update_shard = true;

          if (!update_shard()) {
            LOG(INFO) << "update shard failed!";
          }
          auto_shard.clear();
          is_done_shard_ = true;
          is_in_update_shard = false;

          LOG(INFO) << "update ps_shard one time, update_time: " << auto_shard.update_time();
          lk_shard.unlock();
          cv_shard_.notify_all();
        } else {
          std::unique_lock<std::mutex> lk_shard(mu_shard_);
          cv_shard_.wait(lk_shard, [this] { return is_done_shard_; });
        }
      } else if (auto_shard.is_finish()) {
        if (thread_id == 0) {
          if (!auto_shard.is_already_save()) {
            auto_shard.save_new_shard(train_config_->dirname(), train_config_->model_name());
          }
        }
      }
    }

    uint64_t batch_id = 0;
    // ReadSample
    {
      auto start = std::chrono::steady_clock::now();
      rpc::TensorResponse response;
      auto hdl = rpc_client_->ReadSampleAsync(hub_eps_[idx], &response);
      if (!hdl->Wait()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        continue;
      }

      ReadSampleOption option;
      response.meta().options().UnpackTo(&option);
      if (option.over()) {
        set_hub_over(idx);
        continue;
      }
      if (option.need_wait()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        continue;
      }

      feature->batchid_tensor = tensorflow::Tensor(tensorflow::DT_INT64, {1});
      auto batchid_vec = feature->batchid_tensor.vec<tensorflow::int64>();
      batchid_vec(0) = option.batch_id();
      batch_id = option.batch_id();
      feature->label_tensor = response.tensor1();

      last_success = true;
      is_normal = true;

      auto& tmp_dense = response.tensor2();
      if (tmp_dense.dtype() == tensorflow::DT_BFLOAT16) {
        feature->dense_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tmp_dense.shape());
        tensorflow::BFloat16ToFloat(tmp_dense.flat<tensorflow::bfloat16>().data(),
                                    feature->dense_tensor.flat<float>().data(), tmp_dense.NumElements());
      } else {
        feature->dense_tensor = tmp_dense;
      }

      feature->debug_info = tensorflow::Tensor(tensorflow::DT_STRING, {train_config_->batch_size()});
      if (train_config_->debug_info().size() > 0) {
        auto debug_data = feature->debug_info.flat<std::string>();
        for (size_t i = 0; i < train_config_->batch_size(); i++) {
          debug_data(i) = option.debug_info(i);
        }
      }

      auto end = std::chrono::steady_clock::now();
      monitor::RunStatus::Instance()->PushTime(
          monitor::kOpsReadSample,
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }
    // type && dim check
    if (feature->label_tensor.dim_size(0) != label_size || feature->label_tensor.dim_size(1) != batch_size) {
      LOG(WARNING) << "Trainer label tensor shape not match, trainer_id: " << trainer_id_
                   << ",  label shape: " << feature->label_tensor.shape().DebugString()
                   << ", config batch_size: " << batch_size
                   << ", label_size: "  << label_size;
      continue;
    }
    if (feature->dense_tensor.dim_size(0) != batch_size ||
        feature->dense_tensor.dim_size(1) != dense_total_size) {
      LOG(WARNING) << "Trainer dense tensor shape not match, trainer_id: " << trainer_id_
                   << ", dense label shape: " << feature->dense_tensor.shape().DebugString()
                   << ", config batch_size: " << batch_size
                   << ", dense_total_size: " << dense_total_size;
      continue;
    }

    // EmbedddingLookup
    {
      auto start = std::chrono::steady_clock::now();
      // 获取变量拓扑关系
      int32_t sum = 0;
      std::unordered_map<std::string, std::vector<std::string>> ps_var_names;
      std::unordered_map<std::string, std::vector<int32_t>> ps_var_size;
      std::unordered_map<std::string, std::vector<int32_t>> ps_var_idx;
      std::unordered_map<std::string, int32_t> ps_var_total_size;
      for (size_t i = 0; i < input_sparse.size(); ++i) {
        auto& sparse_fc = input_sparse[i];
        auto& emb_table = input_sparse_emb_table_names[i];
        auto var_dim = input_sparse_dim[i];

        auto shard_eps = train_config_->placement()->GetEmbPlacement(emb_table);
        for (size_t j = 0; j < shard_eps.size(); ++j) {
          auto ep = shard_eps[j];

          ps_var_names[ep].push_back(emb_table);
          ps_var_size[ep].push_back(var_dim);
          ps_var_idx[ep].push_back(i);
          if (ps_var_total_size.find(ep) == ps_var_total_size.end()) {
            ps_var_total_size[ep] = var_dim;
          } else {
            ps_var_total_size[ep] += var_dim;
          }
        }
        sum += var_dim;
      }

      std::vector<rpc::RpcHandlePtr> hdls;
      std::unordered_map<std::string, rpc::TensorResponse> responses;
      for (auto& it : ps_var_names) {
        auto& ep = it.first;
        auto& varnames = it.second;
        auto& var_idx = ps_var_idx[ep];
        auto& var_size = ps_var_size[ep];

        std::string ps_join_varname = absl::StrJoin(varnames, ",");

        EmbeddingLookupOption option;
        option.set_batch_size(batch_size);
        option.set_is_pretrain(train_config_->is_pretrain());
        option.set_work_mode((sniper::WorkMode)work_mode_);
        *(option.mutable_field_idx()) = {var_idx.begin(), var_idx.end()};
        *(option.mutable_field_dim()) = {var_size.begin(), var_size.end()};

        auto hdl =
            rpc_client_->EmbeddingLookupAsync(ep, batch_id, ps_join_varname, option, tensorflow::Tensor(),
                                              tensorflow::Tensor(), &(responses[ep]));
        hdls.push_back(hdl);
      }

      bool lookup_ok = true;
      for (auto& h : hdls) {
        if (!h->Wait()) {
          LOG(WARNING) << "Trainer embedding lookup fail, trainer_id: " << trainer_id_
                       << ", batch_id: " << batch_id
                       << ", ep: " << h->ep()
                       << ", errmsg: " << h->errmsg();

          lookup_ok = false;
          break;
        }
      }
      if (!lookup_ok) {
        continue;
      }

      lookup_ok = true;
      for (auto& it : responses) {
        auto& ep = it.first;
        auto& resp = it.second;

        EmbeddingLookupOption option;
        resp.meta().options().UnpackTo(&option);
        if (!option.errmsg().empty()) {
          LOG(WARNING) << "Trainer embedding lookup fail, trainer_id: " << trainer_id_
                       << ", batch_id: " << batch_id
                       << ", ep: " << ep
                       << ", errmsg: " << option.errmsg();

          lookup_ok = false;
          break;
        }

        if (use_auto_shard_) {
          // 记录响应时间
          if (thread_id == 0 && !auto_shard.is_finish()) {
            auto_shard.add_time_spend(ps_to_index, ps_var_names, ep, option);
          }
        }

        auto& lookup_ret = resp.tensor1();

        auto var_total_size = ps_var_total_size[ep];
        if (lookup_ret.dim_size(0) != batch_size || lookup_ret.dim_size(1) != var_total_size) {
          LOG(WARNING) << "Trainer lookup ret tensor shape not match, trainer_id: " << trainer_id_
                       << ", tensor shape: " << lookup_ret.shape().DebugString()
                       << ", batch_size: " << batch_size
                       << ", total_size: " << var_total_size;

          lookup_ok = false;
          break;
        }

        // dequantization
        tensorflow::Tensor float_lookup_ret;
        if (lookup_ret.dtype() == tensorflow::DT_BFLOAT16) {
          float_lookup_ret = tensorflow::Tensor(tensorflow::DT_FLOAT, lookup_ret.shape());
          tensorflow::BFloat16ToFloat(lookup_ret.flat<tensorflow::bfloat16>().data(),
                                      float_lookup_ret.flat<float>().data(), lookup_ret.NumElements());
        } else {
          float_lookup_ret = lookup_ret;
        }
        auto float_lookup_flat = float_lookup_ret.flat<float>();

        if (float_lookup_ret.dim_size(0) != batch_size || float_lookup_ret.dim_size(1) != var_total_size) {
          LOG(WARNING) << "Trainer float lookup ret tensor shape not match, trainer_id: " << trainer_id_
                       << ", tensor shape: " << float_lookup_ret.shape().DebugString()
                       << ", batch_size: " << batch_size
                       << ", total_size: " << var_total_size;

          lookup_ok = false;
          break;
        }
        if (!lookup_ok) {
          break;
        }

        // 拆包 (ep纬度拆到 var纬度)
        auto& var_names = ps_var_names[ep];
        auto& var_size = ps_var_size[ep];
        auto& var_idx = ps_var_idx[ep];
        int idx = 0;
        for (size_t i = 0; i < var_names.size(); ++i) {
          auto& name = var_names[i];
          auto& size = var_size[i];
          auto& var_lookup_ret = feature->embedding_tensor[var_idx[i]];

          auto var_lookup_flat = var_lookup_ret.flat<float>();
          auto shard_eps = train_config_->placement()->GetEmbPlacement(name);
          auto shard_num = shard_eps.size();

          if (shard_num == 1) {
            for (int j = 0; j < batch_size; ++j) {
              std::copy_n(float_lookup_flat.data() + j * var_total_size + idx, size,
                          var_lookup_flat.data() + j * size);
            }
          } else {
            for (int j = 0; j < batch_size; ++j) {
              Sum(float_lookup_flat.data() + j * var_total_size + idx,
                  var_lookup_flat.data() + j * size,
                  size);
            }
          }
          idx += size;
        }
        if (!lookup_ok) {
          break;
        }
      }
      if (!lookup_ok) {
        continue;
      }

      auto end = std::chrono::steady_clock::now();
      monitor::RunStatus::Instance()->PushTime(
          monitor::kOpsEmbeddingLookup,
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    // type && dim check
    for (size_t i = 0; i < input_sparse_dim.size(); ++i) {
      if (feature->embedding_tensor[i].dim_size(0) != batch_size ||
          feature->embedding_tensor[i].dim_size(1) != input_sparse_dim[i]) {
        LOG(WARNING) << "Trainer feature lookup ret shape not match"
                     << ", trainer_id: " << trainer_id_
                     << ", input_sparse: " << input_sparse[i]
                     << ", tensor shape: " << feature->embedding_tensor[i].shape().DebugString()
                     << ", batch_size: " << batch_size
                     << ", dim: " << input_sparse_dim[i];

        continue;
      }
    }

    // push to queue
    {
      std::unique_lock<std::mutex> lock(mu_);
      while (queue_.size() >= queue_size_ && !finished_) {
        cv_.wait_for(lock, std::chrono::milliseconds(1));
      }

      if (!finished_) {
        queue_.push(feature);
        lock.unlock();
        cv_.notify_one();
      }
    }

    if (train_config_->debug_offline()) {
      SemaphoreLoop::GetInstance().Release(1);
    }
    feature = NewFeatureColumn();
  }

  if (train_config_->debug_offline()) {
    if (is_normal) {
      LOG(INFO) << "ReleaseAll";
      SemaphoreLoop::GetInstance().ReleaseAll(5);
    }
  }
}

bool FeedQueue::is_over() {
  for (auto b : hub_over_) {
    if (!b) {
      return false;
    }
  }
  return true;
}

int32_t FeedQueue::get_next_hub() {
  std::unique_lock<std::mutex> lk(mu_);
  for (size_t i = 0; i < hub_over_.size(); ++i) {
    int can_use = (hub_idx_ + i) % hub_over_.size();
    if (!hub_over_[can_use]) {
      ++hub_idx_;
      hub_idx_ %= hub_over_.size();
      return can_use;
    }
  }
  LOG(WARNING) << "Trainer get next hub fail, trainer_id_: " << trainer_id_;
  return -1;
}

void FeedQueue::set_hub_over(int32_t idx) {
  std::unique_lock<std::mutex> lk(mu_);
  LOG(WARNING) << "hub is over!!! hub idx: " << idx;
  hub_over_[idx] = true;
}

bool FeedQueue::update_shard() {
  auto& auto_shard = AutoShard::instance();
  std::vector<std::vector<size_t>> new_shard = auto_shard.compute_shard();
  train_config_->placement()->UpdateSparsePlacement(new_shard);

  const std::vector<std::vector<size_t>>& new_alloc_shard = auto_shard.new_alloc_shard();
  const std::vector<std::vector<size_t>>& origin_ps_shard = auto_shard.origin_ps_shard();

  const auto& ps_eps = train_config_->ps_eps();

  std::vector<rpc::RpcHandlePtr> hdls;
  std::vector<std::string> table_names;

  // update ps
  for (size_t field = 0; field < new_alloc_shard.size(); field++) {
    if (new_alloc_shard[field].size() == 0) {
      continue;
    }

    for (size_t i = 0; i < new_alloc_shard[field].size(); i++) {
      size_t ps_index = new_alloc_shard[field][i];
      if (ps_index >= ps_eps.size()) {
        LOG(INFO) << "out of range, ps_index: " << ps_index << ", ps_eps.size(): " << ps_eps.size();
        continue;
      }
      const auto& ps_name = ps_eps[ps_index];

      std::string varname = std::string("embedding_") + std::to_string(field);
      absl::optional<CreateOption> option =
          get_create_option(varname, train_config_, i, new_alloc_shard[field].size());
      if (option) {
        option->set_delete_var(false);
        auto h = rpc_client_->CreateAsync(ps_name, varname, *option, 180000);
        hdls.push_back(h);
        table_names.push_back(varname);
      }
    }
  }

  for (size_t i = 0; i < hdls.size(); ++i) {
    auto& h = hdls[i];
    auto& table_name = table_names[i];
    if (!h->Wait()) {
      LOG(WARNING) << "recreate embedding_table[" << table_name << "] on ps[" << h->ep()
                   << "] fail. errmsg=" << h->errmsg();
      return false;
    } else {
      LOG(INFO) << "recreate embedding_table[" << table_name << "] on ps[" << h->ep() << "] success.";
    }
  }

  hdls.clear();
  // update hub
  for (size_t i = 0; i < hub_eps_.size(); i++) {
    const std::string& hub_name = hub_eps_[i];
    UpdateHubShardOption hub_option;

    hub_option.set_hub_idx(i);
    auto mutable_shard = hub_option.mutable_ps_shard();
    for (size_t field = 0; field < new_shard.size(); field++) {
      auto last = mutable_shard->add_value();
      for (size_t ps_index : new_shard[field]) {
        last->add_value(ps_index);
      }
    }

    auto h = rpc_client_->UpdateHubShardAsync(hub_name, hub_option);
    hdls.push_back(h);
  }

  for (size_t i = 0; i < hdls.size(); ++i) {
    auto& h = hdls[i];
    if (!h->Wait()) {
      LOG(WARNING) << "update hub shard failed! hub_name: " << h->ep() << ", errmsg=" << h->errmsg();
      return false;
    } else {
      LOG(WARNING) << "update hub shard success! hub_name: " << h->ep();
    }
  }

  hdls.clear();
  table_names.clear();
  // delete var
  for (size_t field = 0; field < new_alloc_shard.size(); field++) {
    if (new_alloc_shard[field].size() == 0) {
      continue;
    }
    if (field >= origin_ps_shard.size()) {
      LOG(INFO) << "out of range, field: " << field
                << ", origin_ps_shard.size(): " << origin_ps_shard.size();
      continue;
    }

    for (size_t i = 0; i < origin_ps_shard[field].size(); i++) {
      size_t ps_index = origin_ps_shard[field][i];
      if (auto_shard.is_in_vector(ps_index, new_alloc_shard[field])) {
        continue;
      }

      if (ps_index >= ps_eps.size()) {
        LOG(INFO) << "out of range, ps_index: " << ps_index << ", ps_eps.size(): " << ps_eps.size();
        continue;
      }
      const auto& ps_name = ps_eps[ps_index];
      std::string varname = std::string("embedding_") + std::to_string(field);

      absl::optional<CreateOption> option =
          get_create_option(varname, train_config_, i, origin_ps_shard[field].size());
      if (option) {
        option->set_delete_var(true);
        auto h = rpc_client_->CreateAsync(ps_name, varname, *option, 180000);
        hdls.push_back(h);
        table_names.push_back(varname);
      }
    }
  }

  for (size_t i = 0; i < hdls.size(); ++i) {
    auto& h = hdls[i];
    if (!h->Wait()) {
      if (i < table_names.size()) {
        LOG(WARNING) << "delete var failed! ps_name: " << h->ep() << ", varname: " << table_names[i]
                     << ", errmsg=" << h->errmsg();
      }
      return false;
    } else {
      if (i < table_names.size()) {
        LOG(WARNING) << "delete var success! ps_name: " << h->ep() << ", varname: " << table_names[i];
      }
    }
  }

  return true;
}

absl::optional<CreateOption> FeedQueue::get_create_option(const std::string& varname,
                                                          TrainConfig* train_config_ptr, int origin_idx,
                                                          int origin_shard_num) {
  if (train_config_ptr == nullptr) {
    return absl::nullopt;
  }

  const auto& emb_tables = train_config_ptr->emb_tables();
  auto it = emb_tables.find(varname);
  if (it == emb_tables.end()) {
    return absl::nullopt;
  }

  const auto& table = it->second;

  CreateOption option;
  option.set_emb_size(table.dim());
  option.set_capacity(static_cast<uint32_t>(table.capacity() / origin_shard_num));
  option.set_type(VAR_EMBEDDING);
  option.mutable_fields()->CopyFrom(table.fields());
  option.set_hit_limit(table.hit_limit());
  option.set_hash_size(table.hash_bucket_size());

  option.set_use_param_vector(train_config_ptr->use_param_vector());

  // TODO for test default opt is adam
  option.set_optimizer(train_config_ptr->optimizer());
  option.set_beta1(train_config_ptr->beta1());
  option.set_beta2(train_config_ptr->beta2());
  option.set_embedding_lr(train_config_ptr->embedding_lr());
  option.set_embedding_eps(train_config_ptr->embedding_eps());

  option.set_feature_inclusion_freq(train_config_ptr->feature_inclusion_freq());

  option.set_shard_idx(origin_idx);
  option.set_shard_num(origin_shard_num);

  return absl::make_optional(option);
}

}  // namespace ops
}  // namespace sniper
