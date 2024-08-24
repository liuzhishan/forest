#pragma once

#include <dlfcn.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "glog/logging.h"
#include "absl/types/optional.h"
#include "core/operators/kernels/train_config.h"
#include "core/rpc/rpc_client.h"
#include "trainer/core/util/placement/auto_shard.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"

namespace sniper {
namespace ops {

constexpr size_t DEBUG_MAX = 4;

struct FeatureColumn {
  explicit FeatureColumn(int32_t batch_size,
                         int32_t sparse_field_count,
                         const std::vector<int32_t>& sparse_dim) {
    for (auto& emb_dim : sparse_dim) {
      embedding_tensor.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {batch_size, emb_dim}));
    }

    for (auto& emb : embedding_tensor) {
      emb.flat<float>().setZero();
    }
  }

  // DT_INT64, { 1 }
  tensorflow::Tensor batchid_tensor;

  // DT_INT32, { batch_size, label_size }
  tensorflow::Tensor label_tensor;

  // DT_FLOAT, { batch_size, dense_total_size }
  tensorflow::Tensor dense_tensor;

  // DT_INT64, { batch_size, emb_size }
  std::vector<tensorflow::Tensor> embedding_tensor;

  // debug_info, str, { batch_size }
  tensorflow::Tensor debug_info;
};

std::string join_float(float* p, int32_t n);

class FeedQueue : public tensorflow::ResourceBase {
 public:
  explicit FeedQueue(const std::string& conf_file,
                     int32_t trainer_id,
                     int32_t work_mode);

  ~FeedQueue() {
    finished_ = true;
    ready_to_consume_ = true;
    for (auto p_thread : threads_) {
      p_thread->join();
    }
    stat_bg_->join();
    delete stat_bg_;

    if (queue_.empty()) {
      LOG(INFO) << "Trainer " << trainer_id_ << " feed queue destroyed.";
    } else {
      LOG(INFO) << "Trainer " << trainer_id_
                << " feed queue destroyed, but queue is not empty, queue_.size(): "
                << queue_.size();
    }
  }

  bool feed(FeatureColumn*& feature, bool* over);
  void reset();

  std::string DebugString() const { return ""; }
  absl::optional<CreateOption> get_create_option(const std::string& varname, TrainConfig* train_config_ptr,
                                                 int origin_idx, int origin_shard_num);

 private:
  void init();
  void consume(int thread_id);
  FeatureColumn* NewFeatureColumn() {
    return new FeatureColumn(train_config_->batch_size(), train_config_->input_sparse().size(),
                             train_config_->input_sparse_dim());
  }

  // plz locked
  bool is_over();
  int32_t get_next_hub();
  void set_hub_over(int32_t idx);

  bool update_shard();

  std::string conf_file_;
  int32_t trainer_id_;
  TrainConfig* train_config_;
  int32_t work_mode_;
  int32_t parallel_ = 10;
  size_t queue_size_ = 8;
  std::vector<std::string> hub_eps_;

  std::vector<std::thread*> threads_;

  bool finished_ = false;
  bool over_ = false;
  std::queue<FeatureColumn*> queue_;
  mutable std::mutex mu_;
  std::condition_variable cv_;

  std::vector<bool> hub_over_;  // hub server queue is over?
  int32_t hub_idx_ = 0;

  std::thread* stat_bg_;

  rpc::RPCClient* rpc_client_;

  std::atomic<bool> ready_to_consume_;

  bool use_auto_shard_ = false;
  bool is_done_shard_ = true;
  std::condition_variable cv_shard_;
  std::mutex mu_shard_;
};

}  // namespace ops
}  // namespace sniper
