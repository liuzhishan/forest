#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/types/optional.h"
#include "include/json/json.h"
#include "core/proto/meta.pb.h"
#include "core/proto/train_config.pb.h"
#include "core/util/placement/feature_placement.h"
#include "spdlog/spdlog.h"

namespace sniper {
namespace ops {

struct NetworkDesc {
  std::vector<std::string> input_dense;
  std::vector<std::string> input_sparse;
};

class TrainConfig {
 public:
  ~TrainConfig() { delete placement_; }
  TrainConfig(const TrainConfig&) = delete;
  TrainConfig(TrainConfig&&) = delete;
  TrainConfig& operator=(const TrainConfig&) = delete;
  TrainConfig& operator=(TrainConfig&&) = delete;

  static TrainConfig* GetInstance(const std::string& config_path,
                                  int32_t trainer_id) {
    static TrainConfig c(config_path, trainer_id);
    return &c;
  }

  static void ConvertConfigToPb(TrainConfig* config, StartSampleOption& option);

  const std::vector<std::string>& hub_eps() const { return hub_eps_; }
  const std::vector<std::string>& ps_eps() const { return ps_eps_; }
  const std::vector<std::string>& emb_table_names() const {
    return emb_table_names_;
  }
  const std::unordered_map<std::string, std::vector<std::string>>& ps_shard()
      const {
    return ps_shard_;
  }
  const std::map<std::string, EmbeddingTable>& emb_tables() const {
    return emb_tables_;
  }
  int32_t batch_size() { return batch_size_; }
  int32_t dense_total_size() { return dense_total_size_; }
  int32_t sparse_total_size() { return sparse_total_size_; }
  int32_t label_size() { return label_size_; }

  const std::vector<std::string>& input_dense() const { return input_dense_; }
  const std::vector<std::string>& input_sparse() const { return input_sparse_; }
  const std::vector<int32_t>& input_dense_dim() const {
    return input_dense_dim_;
  }
  const std::vector<int32_t>& input_sparse_dim() const {
    return input_sparse_dim_;
  }
  const std::unordered_map<std::string, NumericColumn>& dense_fcs() const {
    return dense_fcs_;
  }
  const std::unordered_map<std::string, std::vector<EmbeddingColumn>>&
  sparse_fcs() const {
    return sparse_fcs_;
  }
  const std::unordered_map<std::string, SeqColumn>& seq_fcs() const {
    return seq_fcs_;
  }
  FeaturePlacement* placement() { return placement_; }

  const Basic& basic() const { return basic_; }

  std::string model_name() { return model_name_; }
  std::string tab() { return tab_; }
  std::string item_type() { return item_type_; }
  std::string item_filter() { return item_filter_; }
  std::string label_extractor() { return label_extractor_; }
  std::string hash_type() { return hash_type_; }
  bool train_hash() { return train_hash_; }
  bool use_weight() { return use_weight_; }
  bool debug() { return debug_; }
  bool enable_format_opt() { return enable_format_opt_; }

  std::string aim() { return aim_; }
  float sample_rate() { return sample_rate_; }
  float neg_sample_rate() { return neg_sample_rate_; }

  std::string topic() { return topic_; };
  std::string group_id() { return group_id_; }
  std::string consumer_user_param() { return consumer_user_param_; }
  bool need_batch() { return need_batch_; }
  std::string checkpoint_version() { return checkpoint_version_; }
  int32_t ckp_save_btq_incr_sparse_step() {
    return ckp_save_btq_incr_sparse_step_;
  }

  int64_t btq_push_limit_total() { return btq_push_limit_total_; }
  bool use_hash_distribute() { return use_hash_distribute_; }

  std::string embedding_table_store_type() {
    return embedding_table_store_type_;
  }

  float eps() { return eps_; }
  float l2() { return l2_; }
  float decay() { return decay_; }
  int32_t version() { return version_; }
  bool use_freq_scale() { return use_freq_scale_; }

  std::string debug_info() { return debug_info_; }
  bool kafka_train() { return kafka_train_; }
  bool debug_offline() { return debug_ && !kafka_train_; }
  bool is_pretrain() { return is_pretrain_; }
  int message_obsolete_seconds() { return message_obsolete_seconds_; }

  float embedding_lr() const { return embedding_lr_; }
  float embedding_eps() const { return embedding_eps_; }
  float beta1() const { return beta1_; }
  float beta2() const { return beta2_; }
  std::string optimizer() const { return optimizer_; }
  int32_t btq_topic_num() const { return btq_topic_num_; }

  std::string dragon_pipeline_str() const { return dragon_pipeline_str_; }
  int queue_str_size() const { return queue_str_size_; }
  bool dragonfly() const { return dragonfly_; }
  int neg_label() const { return neg_label_; }
  bool fake_missed_feature() const { return fake_missed_feature_; }
  bool batched_sample() const { return batched_sample_; }
  bool is_input_one_column() const { return is_input_one_column_; }
  std::string kafka_tag() const { return kafka_tag_; }
  int feed_queue_max_size() const { return feed_queue_max_size_; }
  bool shuffle() const { return shuffle_; }
  int shuffle_size() const { return shuffle_size_; }
  bool parallel_eq_cpu_core() const { return parallel_eq_cpu_core_; }
  int feed_worker() const { return feed_worker_; }

  int hub_stream_num() const { return hub_stream_num_; }
  int hub_stream_size() const { return hub_stream_size_; }
  int hub_train_log_processor_num() const { return hub_train_log_processor_num_; }
  int hub_train_log_processor_size() const { return hub_train_log_processor_size_; }
  int hub_feed_node_num() const { return hub_feed_node_num_; }
  int hub_feed_node_size() const { return hub_feed_node_size_; }
  int feature_inclusion_freq() const { return feature_inclusion_freq_; }
  bool is_kafka_feature() const { return is_kafka_feature_; }
  int dragon_queue_size() const { return dragon_queue_size_; }
  bool use_param_vector() const { return use_param_vector_; }

  bool use_auto_shard() const { return use_auto_shard_; }
  int top_ps() const { return top_ps_; }
  int top_field() const { return top_field_; }
  int field_shard_limit() const { return field_shard_limit_; }
  int update_shard_limit() const { return update_shard_limit_; }
  int step_limit() const { return step_limit_; }
  bool is_move_shard() const { return is_move_shard_; }

  const std::string& dirname() const { return (dirname_); }

  const std::unordered_map<std::string, size_t>& ps_to_index() const { return (ps_to_index_); }
  const std::vector<std::vector<size_t>>& ps_shard_by_field() const { return (ps_shard_by_field_); }

 private:
  explicit TrainConfig(const std::string& config_path, int32_t trainer_id)
      : config_path_(config_path),
        trainer_id_(trainer_id),
        checkpoint_version_("3"),
        btq_push_limit_total_(0),
        use_hash_distribute_(false) {
    if (!parse()) {
      SPDLOG_ERROR("parse train config fail. i will quit...");
      abort();
    }
    std::vector<EmbeddingTable> vars;
    for (auto& it : emb_tables_) {
      vars.push_back(it.second);
    }
    placement_ = new FeaturePlacement(vars, ps_eps_, ps_shard_, emb_load_);
  }
  bool parse();
  bool parse_basic(Json::Value const& value);
  bool parse_clusterspec(Json::Value const& value);
  bool parse_network(Json::Value const& value);
  bool parse_feature_column(Json::Value const& value);
  bool parse_embedding_table(Json::Value const& value);
  bool parse_sample(Json::Value const& value);
  bool parse_trainer(Json::Value const& value);
  bool post_parse();
  // bool parse_prealloc_ps(Json::Value const& value);

  bool read_file(const char* path, Json::Value* root);

  std::string config_path_;
  int32_t trainer_id_;

 private:
  // basic
  Basic basic_;

  // clusterspec
  std::vector<std::string> hub_eps_;
  std::vector<std::string> ps_eps_;
  std::unordered_map<std::string, size_t> ps_to_index_;
  std::vector<std::string> trainer_eps_;
  // var -> shards ep
  std::unordered_map<std::string, std::vector<std::string>> ps_shard_;
  std::vector<std::vector<size_t>> ps_shard_by_field_;
  // var -> load
  std::unordered_map<std::string, float> emb_load_;

  // network
  std::vector<std::string> input_dense_;
  std::vector<std::string> input_sparse_;
  std::vector<std::string> emb_table_names_;
  std::unordered_map<std::string, NetworkDesc>
      loss_funcs_;  // loss_name -> loss newtwork
  std::vector<int32_t> input_dense_dim_;
  std::vector<int32_t> input_sparse_dim_;

  // feature column
  std::unordered_map<std::string, NumericColumn> dense_fcs_;
  std::unordered_map<std::string, std::vector<EmbeddingColumn>> sparse_fcs_;
  std::unordered_map<std::string, SeqColumn> seq_fcs_;

  // embedding table
  std::map<std::string, EmbeddingTable> emb_tables_;

  // sample
  int32_t batch_size_;
  int32_t dense_total_size_;
  int32_t sparse_total_size_;
  int32_t label_size_;

  // emb placement
  FeaturePlacement* placement_;

  std::string model_name_;
  std::string tab_;
  std::string item_type_;
  std::string item_filter_;
  std::string label_extractor_;
  std::string hash_type_;
  bool train_hash_;
  bool use_weight_;
  bool debug_;

  std::string aim_;
  float sample_rate_;
  float neg_sample_rate_;

  std::string topic_;
  std::string group_id_;
  std::string consumer_user_param_;

  bool enable_format_opt_;

  bool need_batch_;
  std::string checkpoint_version_;
  int32_t btq_push_limit_total_;
  bool use_hash_distribute_;
  int32_t ckp_save_btq_incr_sparse_step_;

  std::string embedding_table_store_type_;

  float eps_ = 1e-8;
  float decay_ = 0.0;
  float l2_ = 0.0;
  int version_ = 2;
  bool use_freq_scale_ = false;

  std::string debug_info_;
  bool kafka_train_ = false;
  bool debug_offline_ = false;
  bool is_pretrain_ = false;
  int message_obsolete_seconds_ = 0;

  // optional
  float embedding_lr_ = 0.0;
  float embedding_eps_ = 0.0;

  float beta1_ = 0.0;
  float beta2_ = 0.0;
  std::string optimizer_;
  int32_t btq_topic_num_ = 0;

  std::string dragon_pipeline_str_ = "";
  int queue_str_size_ = 0;
  bool dragonfly_ = false;
  int neg_label_ = 1;
  bool fake_missed_feature_ = false;
  bool batched_sample_ = false;
  bool is_input_one_column_ = true;
  std::string kafka_tag_ = "";
  int feed_queue_max_size_ = 64;
  bool shuffle_ = false;
  int shuffle_size_ = 20000;
  bool parallel_eq_cpu_core_ = false;
  int feed_worker_ = 0;

  int hub_stream_num_ = 0;
  int hub_stream_size_ = 0;
  int hub_train_log_processor_num_ = 0;
  int hub_train_log_processor_size_ = 0;
  int hub_feed_node_num_ = 0;
  int hub_feed_node_size_ = 0;
  int feature_inclusion_freq_ = 0;
  bool is_kafka_feature_ = false;
  int dragon_queue_size_ = 300;
  bool use_param_vector_ = false;

  // auto_shard
  bool use_auto_shard_ = false;
  int top_ps_ = 1;
  int top_field_ = 1;
  int field_shard_limit_ = 2;
  int update_shard_limit_ = 1;
  int step_limit_ = 1000;
  bool is_move_shard_ = true;

  std::string dirname_;
};

}  // namespace ops
}  // namespace sniper
