#include <cctype>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "core/base/util.h"
#include "core/operators/kernels/train_config.h"
#include "absl/types/optional.h"

namespace sniper {
namespace ops {

void TrainConfig::ConvertConfigToPb(TrainConfig* config,
                                    StartSampleOption& option) {
  auto& ps_eps = config->ps_eps();
  auto& hub_eps = config->hub_eps();
  auto& input_dense = config->input_dense();
  auto& input_sparse = config->input_sparse();
  auto batch_size = config->batch_size();
  auto dense_total_size = config->dense_total_size();
  auto label_size = config->label_size();

  *option.mutable_ps_eps() = {ps_eps.begin(), ps_eps.end()};

  *(option.mutable_feature_list()->mutable_dense_class_names()) = {
      input_dense.begin(), input_dense.end()};
  *(option.mutable_feature_list()->mutable_sparse_class_names()) = {
      input_sparse.begin(), input_sparse.end()};
  option.set_batch_size(batch_size);
  option.set_dense_total_size(dense_total_size);
  option.set_label_size(label_size);
  option.set_enable_format_opt(config->enable_format_opt());
  option.set_use_hash_distribute(config->use_hash_distribute());

  for (auto& it : config->emb_tables()) {
    option.add_emb_tables()->CopyFrom(it.second);
  }

  for (const auto& hub : hub_eps) {
    option.add_hub_eps(hub);
  }

  option.set_model_name(config->model_name());
  option.set_batch_size(config->batch_size());

  for (size_t i = 0; i < config->input_sparse().size(); i++) {
    auto sparse_name = config->input_sparse()[i];
    option.mutable_feature_list()->mutable_sparse_field_count()->Add(
        config->input_sparse_dim()[i]);

    int prefix = -1;
    int slot = -1;
    std::string emb_table_name = "";
    auto sparse_it = config->sparse_fcs().find(sparse_name);
    auto seq_it = config->seq_fcs().find(sparse_name);
    auto add_func = [&]() -> void {
      if (config->emb_tables().find(emb_table_name) ==
          config->emb_tables().end()) {
        SPDLOG_ERROR("emb table[{}] not find", emb_table_name);
        return;
      }

      int64_t bucket_size =
          config->emb_tables().find(emb_table_name)->second.hash_bucket_size();
      option.mutable_feature_list()->mutable_sparse_hash_size()->Add(
          bucket_size);
      option.mutable_feature_list()->mutable_sparse_field_index()->Add(i);

      auto field_info =
          option.mutable_feature_list()->mutable_field_list()->Add();
      field_info->set_class_name(sparse_name);
      field_info->set_prefix(prefix);
      field_info->set_index(i);
      field_info->set_size(bucket_size);
      field_info->set_valid(true);
      field_info->set_slot(slot);
      auto* tmp =
          option.mutable_feature_list()->mutable_sparse_emb_table()->Add();
      *tmp = emb_table_name;
    };
    if (sparse_it == config->sparse_fcs().end() &&
        seq_it == config->seq_fcs().end()) {
      SPDLOG_ERROR("sparse feature[{}] not find", sparse_name);
      return;
    } else if (sparse_it != config->sparse_fcs().end()) {
      for (auto& v : sparse_it->second) {
        if (i != static_cast<size_t>(v.sparse_index())) {
          continue;
        }
        emb_table_name = v.emb_table();
        prefix = v.prefix();
        add_func();
      }
    } else {
      emb_table_name = seq_it->second.share_emb_table();
      slot = seq_it->second.share_slot();
      add_func();
    }
  }

  for (size_t i = 0; i < config->input_dense().size(); i++) {
    auto dense_name = config->input_dense()[i];
    option.mutable_feature_list()->mutable_dense_field_count()->Add(
        config->input_dense_dim()[i]);
  }

  if (option.src() == klearn::SRC_HDFS) {
    // some hdfs parameters
    option.mutable_hdfs_src()->set_train_hash(config->train_hash());
  } else if (option.src() == klearn::SRC_KAFKA) {
    option.mutable_kafka_src()->set_topic(config->topic());
    option.mutable_kafka_src()->set_group_id(config->group_id());
    option.mutable_kafka_src()->set_consumer_user_param(
        config->consumer_user_param());
    option.mutable_kafka_src()->set_kafka_tag(config->kafka_tag());
  } else if (option.src() == klearn::SRC_BTQ) {
    option.mutable_kafka_src()->set_topic(config->topic());
  } else if (option.src() == klearn::SRC_DRAGON) {
    option.mutable_hdfs_src()->set_train_hash(config->train_hash());
  } else {
    SPDLOG_ERROR("wrong src: {}", option.src());
    return;
  }

  for (auto& one_ps : config->ps_eps()) {
    auto tmp = option.mutable_shard_addr()->Add();
    *tmp = one_ps;
  }
  option.set_dense_total_size(config->dense_total_size());

  option.set_tab(config->tab());
  option.set_item_type(config->item_type());
  option.set_item_filter(config->item_filter());
  option.set_label_extractor(config->label_extractor());
  option.set_hash_type(config->hash_type());
  option.set_use_weight(config->use_weight());
  option.set_debug(config->debug());
  option.set_aim(config->aim());
  option.set_sample_rate(config->sample_rate());
  option.set_neg_sample_rate(config->neg_sample_rate());
  option.set_need_batch(config->need_batch());
  option.set_checkpoint_version(config->checkpoint_version());
  option.set_enable_format_opt(config->enable_format_opt());
  option.set_debug_info(config->debug_info());
  option.set_message_obsolete_seconds(config->message_obsolete_seconds());

  option.set_dragon_pipeline_str(config->dragon_pipeline_str());
  option.set_queue_str_size(config->queue_str_size());
  option.set_dragonfly(config->dragonfly());
  option.set_neg_label(config->neg_label());
  option.set_fake_missed_feature(config->fake_missed_feature());
  option.set_batched_sample(config->batched_sample());
  option.set_is_input_one_column(config->is_input_one_column());
  option.set_feed_queue_max_size(config->feed_queue_max_size());
  option.set_shuffle(config->shuffle());
  option.set_shuffle_size(config->shuffle_size());
  option.set_parallel_eq_cpu_core(config->parallel_eq_cpu_core());

  option.set_hub_stream_num(config->hub_stream_num());
  option.set_hub_stream_size(config->hub_stream_size());
  option.set_hub_train_log_processor_num(config->hub_train_log_processor_num());
  option.set_hub_train_log_processor_size(config->hub_train_log_processor_size());
  option.set_hub_feed_node_num(config->hub_feed_node_num());
  option.set_hub_feed_node_size(config->hub_feed_node_size());
  option.set_is_kafka_feature(config->is_kafka_feature());
  option.set_dragon_queue_size(config->dragon_queue_size());
}

bool TrainConfig::parse() {
  Json::Value value;
  auto ret = read_file(config_path_.c_str(), &value);
  if (!ret) return false;

  SPDLOG_INFO("start parse train config!");

  if (!parse_basic(value)) {
    SPDLOG_ERROR("parse basic fail");
    return false;
  }
  SPDLOG_INFO("start parse train config! parse basic done");
  if (!parse_clusterspec(value)) {
    SPDLOG_ERROR("parse clusterspec fail");
    return false;
  }
  SPDLOG_INFO("start parse train config! parse cluster done");
  if (!parse_network(value)) {
    SPDLOG_ERROR("parse network fail");
    return false;
  }
  SPDLOG_INFO("start parse train config! parse network done");
  if (!parse_feature_column(value)) {
    SPDLOG_ERROR("parse feature_column fail");
    return false;
  }
  if (!parse_embedding_table(value)) {
    SPDLOG_ERROR("parse embedding_table fail");
    return false;
  }
  SPDLOG_INFO("start parse train config! parse embedding table done");
  if (!parse_sample(value)) {
    SPDLOG_ERROR("parse sample fail");
    return false;
  }

  // if(!parse_prealloc_ps(value)){
  //   SPDLOG_ERROR("parse pre_alloc ps fail");
  //   return false;
  // }

  if (!parse_trainer(value)) {
    SPDLOG_ERROR("parse trainer fail");
    return false;
  }

  SPDLOG_INFO("start parse train config! parse trainer done");

  if (!post_parse()) {
    SPDLOG_ERROR("post parse check fail");
    return false;
  }

  SPDLOG_INFO("start parse train config! post parse done");
  return true;
}

// bool TrainConfig::parse_prealloc_ps(Json::Value const& value) {
//   //读取预分配的ps数
//   if(!value.isMember("alloc_ps") &&
//       value["alloc_ps"].type() == Json::objectValue){
//     return false;
//   }
//   auto alloc_ps = value["alloc_ps"];
//   Json::Value::Members mems = alloc_ps.getMemberNames();
//   for (auto iter = mems.begin(); iter != mems.end(); iter++) {
//     std::string fc_name = *iter;
//     if (alloc_ps[fc_name].type() == Json::arrayValue &&
//     alloc_ps[fc_name].size() > 0) {
//       std::string s;
//       for (int i = 0; i < int(alloc_ps[fc_name].size()); ++i) {
//         ps_shard_[fc_name].push_back(alloc_ps[fc_name][i].asString());
//         s += alloc_ps[fc_name][i].asString() + ",";
//       }
//       SPDLOG_INFO("ps_shard, key: {}, value: {}", fc_name, s);
//     } else {
//       return false;
//     }
//   }
//   return true;
// }

bool TrainConfig::parse_basic(Json::Value const& value) {
  // TODO(dx)
  return true;
}

bool TrainConfig::parse_clusterspec(Json::Value const& value) {
  if (!(value.isMember("clusterspec") &&
        value["clusterspec"].type() == Json::objectValue)) {
    return false;
  }
  auto value_clusterspec = value["clusterspec"];

  if (value_clusterspec.isMember("trainer") &&
      value_clusterspec["trainer"].type() == Json::arrayValue) {
    int size = value_clusterspec["trainer"].size();
    for (int i = 0; i < size; ++i) {
      trainer_eps_.push_back(value_clusterspec["trainer"][i].asString());
    }
  } else {
    return false;
  }

  if (value_clusterspec.isMember("hub") &&
      value_clusterspec["hub"].type() == Json::arrayValue) {
    int size = value_clusterspec["hub"].size();
    for (int i = 0; i < size; ++i) {
      hub_eps_.push_back(value_clusterspec["hub"][i].asString());
    }
  } else {
    return false;
  }

  if (value_clusterspec.isMember("ps") &&
      value_clusterspec["ps"].type() == Json::arrayValue) {
    int size = value_clusterspec["ps"].size();
    for (int i = 0; i < size; ++i) {
      ps_eps_.push_back(value_clusterspec["ps"][i].asString());
    }

    ps_to_index_.clear();
    for (size_t i = 0; i < ps_eps_.size(); i++) {
      ps_to_index_[ps_eps_[i]] = i;
    }
  } else {
    return false;
  }

  return true;
}

bool TrainConfig::parse_network(Json::Value const& value) {
  // TODO(dx)
  if (!(value.isMember("network") &&
        value["network"].type() == Json::objectValue)) {
    return false;
  }
  auto network_column = value["network"];

  if (!(network_column.isMember("inputs") &&
        network_column["inputs"].type() == Json::objectValue)) {
    return false;
  }
  auto inputs = network_column["inputs"];

  {
    if (!(inputs.isMember("dense") &&
          inputs["dense"].type() == Json::arrayValue)) {
      return false;
    }
    auto dense = inputs["dense"];
    Json::ArrayIndex size = dense.size();
    for (Json::ArrayIndex index = 0; index < size; ++index) {
      input_dense_.push_back(dense[index].asString());
    }
  }

  {
    if (!(inputs.isMember("sparse") &&
          inputs["sparse"].type() == Json::arrayValue)) {
      return false;
    }
    auto sparse = inputs["sparse"];
    Json::ArrayIndex size = sparse.size();
    for (Json::ArrayIndex index = 0; index < size; ++index) {
      input_sparse_.push_back(sparse[index].asString());
    }
  }

  {
    if (!(inputs.isMember("emb_table") &&
          inputs["emb_table"].type() == Json::arrayValue)) {
      return false;
    }
    auto tables = inputs["emb_table"];
    Json::ArrayIndex size = tables.size();
    for (Json::ArrayIndex index = 0; index < size; ++index) {
      emb_table_names_.push_back(tables[index].asString());
    }
  }

  return true;
}

bool TrainConfig::parse_feature_column(Json::Value const& value) {
  if (!(value.isMember("feature_column") &&
        value["feature_column"].type() == Json::arrayValue)) {
    return false;
  }
  auto value_feature_column = value["feature_column"];

  Json::ArrayIndex size = value_feature_column.size();
  for (Json::ArrayIndex index = 0; index < size; ++index) {
    Json::Value mem = value_feature_column[index];
    std::string fc_name;
    std::string fc_class;
    if (mem.isMember("class_name") &&
        mem["class_name"].type() == Json::stringValue) {
      fc_name = mem["class_name"].asString();
    } else {
      return false;
    }
    if (mem.isMember("class") && mem["class"].type() == Json::stringValue) {
      fc_class = mem["class"].asString();
    } else {
      return false;
    }
    if (fc_class == "embedding_column") {
      if (mem.isMember("embedding_column") &&
          mem["embedding_column"].type() == Json::objectValue) {
        Json::Value embedding_column = mem["embedding_column"];
        EmbeddingColumn column;
        if (embedding_column.isMember("emb_table") &&
            embedding_column["emb_table"].type() == Json::stringValue) {
          column.set_emb_table(embedding_column["emb_table"].asString());
        }
        column.set_prefix(mem["attrs"]["prefix"].asInt());
        column.set_sparse_index(mem["sparse_index"].asInt());

        auto& columns = sparse_fcs_[fc_name];
        columns.push_back(column);
      }
    }
    if (fc_class == "seq_column") {
      if (mem.isMember("seq_column") &&
          mem["seq_column"].type() == Json::objectValue) {
        Json::Value seq_column = mem["seq_column"];
        SeqColumn column;
        if (seq_column.isMember("emb_table") &&
            seq_column["emb_table"].type() == Json::stringValue) {
          column.set_share_emb_table(seq_column["emb_table"].asString());
        }
        column.set_share_field(mem["attrs"]["share_field"].asInt());
        column.set_share_slot(mem["attrs"]["share_slot"].asInt());
        column.set_sparse_index(mem["sparse_index"].asInt());

        seq_fcs_[fc_name] = column;
      }
    }
    if (fc_class == "numeric_column") {
      if (mem.isMember("numeric_column") &&
          mem["numeric_column"].type() == Json::objectValue) {
        Json::Value numeric_column = mem["numeric_column"];
        NumericColumn column;
        if (numeric_column.isMember("dim") &&
            numeric_column["dim"].type() == Json::intValue) {
          column.set_dim(numeric_column["dim"].asLargestInt());
        }
        dense_fcs_[fc_name] = column;
      }
    }
  }

  return true;
}

bool TrainConfig::parse_embedding_table(Json::Value const& value) {
  if (!(value.isMember("embedding_table") &&
        value["embedding_table"].type() == Json::objectValue)) {
    return false;
  }
  auto value_embedding_table = value["embedding_table"];

  ps_shard_by_field_.clear();
  ps_shard_by_field_.resize(value_embedding_table.size());

  Json::Value::Members mems = value_embedding_table.getMemberNames();
  for (auto iter = mems.begin(); iter != mems.end(); iter++) {
    std::string table_name = *iter;
    Json::Value mem = value_embedding_table[*iter];
    if (mem.type() != Json::objectValue) {
      return false;
    }
    EmbeddingTable table;
    table.set_name(table_name);
    table.set_load(1.0);
    if (mem.isMember("dim") && mem["dim"].type() == Json::intValue) {
      table.set_dim(mem["dim"].asLargestInt());
    }
    if (mem.isMember("capacity") && mem["capacity"].type() == Json::intValue) {
      table.set_capacity(mem["capacity"].asLargestInt());
    }
    if (mem.isMember("hash_bucket_size") &&
        mem["hash_bucket_size"].type() == Json::intValue) {
      table.set_hash_bucket_size(mem["hash_bucket_size"].asLargestInt());
    }
    if (mem.isMember("hash_func") &&
        mem["hash_func"].type() == Json::stringValue) {
      table.set_hash_func(mem["hash_func"].asString());
    }
    if (mem.isMember("load") && (mem["load"].type() == Json::realValue ||
                                 mem["load"].type() == Json::intValue)) {
      table.set_load(mem["load"].asFloat());
      emb_load_[table_name] = mem["load"].asFloat();
    }

    if (mem.isMember("fields") && mem["fields"].type() == Json::arrayValue &&
        mem["fields"].size() > 0) {
      for (int i = 0; i < int(mem["fields"].size()); ++i) {
        table.mutable_fields()->Add(mem["fields"][i].asInt());
      }
    }

    if (mem.isMember("shard") && mem["shard"].type() == Json::arrayValue &&
        mem["shard"].size() > 0) {
      std::string s;
      std::vector<std::string> tmp;
      for (int i = 0; i < int(mem["shard"].size()); ++i) {
        tmp.push_back(mem["shard"][i].asString());
        auto* ptr = table.mutable_ps_shard()->mutable_value()->Add();
        *ptr = mem["shard"][i].asString();

        s += mem["shard"][i].asString() + ",";

        SPDLOG_INFO("shard ps, varname: {}, {}, i: {}", table_name, mem["shard"][i].asString(), i);
        if (absl::optional<int> int_value = find_int_suffix(table_name)) {
          if (*int_value >= ps_shard_by_field_.size()) {
            ps_shard_by_field_.resize(*int_value + 1);
          }

          // 之前必须先读取 ps_eps_ 参数
          auto it_ps_index = ps_to_index_.find(mem["shard"][i].asString());
          if (it_ps_index != ps_to_index_.end()) {
            ps_shard_by_field_[*int_value].push_back(it_ps_index->second);
          } else {
            SPDLOG_INFO("cannot find ps_index: {}", mem["shard"][i].asString());
          }
        }
      }

      ps_shard_[table_name] = tmp;
      SPDLOG_INFO("ps_shard, key: {}, value: {}", table_name, s);
    } else {
      SPDLOG_INFO("sparse feature {} has no shard, exit!", table_name);
      return false;
    }

    if (mem.isMember("hit_limit") &&
        mem["hit_limit"].type() == Json::intValue) {
      table.set_hit_limit(mem["hit_limit"].asLargestInt());
    }

    emb_tables_[table_name] = table;
  }

  return true;
}

bool TrainConfig::parse_sample(Json::Value const& value) {
  if (!(value.isMember("sample") &&
        value["sample"].type() == Json::objectValue)) {
    return false;
  }
  auto value_sample = value["sample"];

  if (value_sample.isMember("batch_size") &&
      value_sample["batch_size"].type() == Json::intValue) {
    batch_size_ = value_sample["batch_size"].asLargestInt();
  }

  if (value_sample.isMember("label_size") &&
      value_sample["label_size"].type() == Json::intValue) {
    label_size_ = value_sample["label_size"].asLargestInt();
  }

  return true;
}

bool TrainConfig::parse_trainer(Json::Value const& value) {
  if (!(value.isMember("trainer") &&
        value["trainer"].type() == Json::objectValue)) {
    return false;
  }

  auto value_trainer = value["trainer"];

  if (value_trainer.isMember("model_name") &&
      value_trainer["model_name"].type() == Json::stringValue) {
    model_name_ = value_trainer["model_name"].asString();
  }
  if (value_trainer.isMember("tab") &&
      value_trainer["tab"].type() == Json::stringValue) {
    tab_ = value_trainer["tab"].asString();
  }
  if (value_trainer.isMember("item_type") &&
      value_trainer["item_type"].type() == Json::stringValue) {
    item_type_ = value_trainer["item_type"].asString();
  }
  if (value_trainer.isMember("item_filter") &&
      value_trainer["item_filter"].type() == Json::stringValue) {
    item_filter_ = value_trainer["item_filter"].asString();
  }
  if (value_trainer.isMember("label_extractor") &&
      value_trainer["label_extractor"].type() == Json::stringValue) {
    label_extractor_ = value_trainer["label_extractor"].asString();
  }
  if (value_trainer.isMember("hash_type") &&
      value_trainer["hash_type"].type() == Json::stringValue) {
    hash_type_ = value_trainer["hash_type"].asString();
  }
  if (value_trainer.isMember("ps_hash") &&
      value_trainer["ps_hash"].type() == Json::intValue) {
    train_hash_ = value_trainer["ps_hash"].asLargestInt() == 1;
  }
  if (value_trainer.isMember("use_weight") &&
      value_trainer["use_weight"].type() == Json::booleanValue) {
    use_weight_ = value_trainer["use_weight"].asBool();
  }
  if (value_trainer.isMember("debug") &&
      value_trainer["debug"].type() == Json::booleanValue) {
    debug_ = value_trainer["debug"].asBool();
  }
  if (value_trainer.isMember("aim") &&
      value_trainer["aim"].type() == Json::stringValue) {
    aim_ = value_trainer["aim"].asString();
  }
  if (value_trainer.isMember("sample_rate") &&
      value_trainer["sample_rate"].type() == Json::realValue) {
    sample_rate_ = value_trainer["sample_rate"].asFloat();
  }
  if (value_trainer.isMember("neg_sample_rate") &&
      value_trainer["neg_sample_rate"].type() == Json::realValue) {
    neg_sample_rate_ = value_trainer["neg_sample_rate"].asFloat();
  }
  if (value_trainer.isMember("enable_format_opt") &&
      value_trainer["enable_format_opt"].type() == Json::booleanValue) {
    enable_format_opt_ = value_trainer["enable_format_opt"].asBool();
  }
  if (value_trainer.isMember("topic") &&
      value_trainer["topic"].type() == Json::stringValue) {
    topic_ = value_trainer["topic"].asString();
  }
  if (value_trainer.isMember("group_id") &&
      value_trainer["group_id"].type() == Json::stringValue) {
    group_id_ = value_trainer["group_id"].asString();
  }
  if (value_trainer.isMember("consumer_user_param") &&
      value_trainer["consumer_user_param"].type() == Json::stringValue) {
    consumer_user_param_ = value_trainer["consumer_user_param"].asString();
  }
  if (value_trainer.isMember("need_batch") &&
      value_trainer["need_batch"].type() == Json::booleanValue) {
    need_batch_ = value_trainer["need_batch"].asBool();
  }
  if (value_trainer.isMember("checkpoint_version") &&
      value_trainer["checkpoint_version"].type() == Json::stringValue) {
    checkpoint_version_ = value_trainer["checkpoint_version"].asString();
  }
  if (value_trainer.isMember("btq_push_limit_total") &&
      value_trainer["btq_push_limit_total"].type() == Json::intValue) {
    btq_push_limit_total_ = value_trainer["btq_push_limit_total"].asInt();
  }
  if (value_trainer.isMember("use_hash_distribute") &&
      value_trainer["use_hash_distribute"].type() == Json::booleanValue) {
    use_hash_distribute_ = value_trainer["use_hash_distribute"].asBool();
  }
  if (value_trainer.isMember("ckp_save_btq_incr_sparse_step") &&
      value_trainer["ckp_save_btq_incr_sparse_step"].type() == Json::intValue) {
    ckp_save_btq_incr_sparse_step_ =
        value_trainer["ckp_save_btq_incr_sparse_step"].asInt();
  }
  if (value_trainer.isMember("embedding_table_store_type") &&
      value_trainer["embedding_table_store_type"].type() == Json::stringValue) {
    embedding_table_store_type_ =
        value_trainer["embedding_table_store_type"].asString();
  }
  if (value_trainer.isMember("eps") &&
      (value_trainer["eps"].type() == Json::realValue ||
       value_trainer["eps"].type() == Json::intValue)) {
    eps_ = value_trainer["eps"].asFloat();
  }
  if (value_trainer.isMember("decay") &&
      (value_trainer["decay"].type() == Json::realValue ||
       value_trainer["decay"].type() == Json::intValue)) {
    decay_ = value_trainer["decay"].asFloat();
  }
  if (value_trainer.isMember("l2") &&
      (value_trainer["l2"].type() == Json::realValue ||
       value_trainer["l2"].type() == Json::intValue)) {
    l2_ = value_trainer["l2"].asFloat();
  }
  if (value_trainer.isMember("version") &&
      value_trainer["version"].type() == Json::intValue) {
    version_ = value_trainer["version"].asInt();
  }
  if (value_trainer.isMember("use_freq_scale") &&
      value_trainer["use_freq_scale"].type() == Json::booleanValue) {
    use_freq_scale_ = value_trainer["use_freq_scale"].asBool();
  }
  if (value_trainer.isMember("debug_info") &&
      value_trainer["debug_info"].type() == Json::stringValue) {
    debug_info_ = value_trainer["debug_info"].asString();
  }
  if (value_trainer.isMember("kafka_train") &&
      value_trainer["kafka_train"].type() == Json::booleanValue) {
    kafka_train_ = value_trainer["kafka_train"].asBool();
  }
  if (value_trainer.isMember("is_pretrain") &&
      value_trainer["is_pretrain"].type() == Json::booleanValue) {
    is_pretrain_ = value_trainer["is_pretrain"].asBool();
  }
  if (value_trainer.isMember("message_obsolete_seconds") &&
      value_trainer["message_obsolete_seconds"].type() == Json::intValue) {
    message_obsolete_seconds_ =
        value_trainer["message_obsolete_seconds"].asInt();
  }

  if (value_trainer.isMember("embedding_lr") &&
      (value_trainer["embedding_lr"].type() == Json::realValue ||
       value_trainer["embedding_lr"].type() == Json::intValue)) {
    embedding_lr_ = value_trainer["embedding_lr"].asFloat();
  }

  if (value_trainer.isMember("embedding_eps") &&
      (value_trainer["embedding_eps"].type() == Json::realValue ||
       value_trainer["embedding_eps"].type() == Json::intValue)) {
    embedding_eps_ = value_trainer["embedding_eps"].asFloat();
  }

  if (value_trainer.isMember("beta1") &&
      (value_trainer["beta1"].type() == Json::realValue ||
       value_trainer["beta1"].type() == Json::intValue)) {
    beta1_ = value_trainer["beta1"].asFloat();
  }
  if (value_trainer.isMember("beta2") &&
      (value_trainer["beta2"].type() == Json::realValue ||
       value_trainer["beta2"].type() == Json::intValue)) {
    beta2_ = value_trainer["beta2"].asFloat();
  }

  if (value_trainer.isMember("optimizer") &&
      value_trainer["optimizer"].type() == Json::stringValue) {
    optimizer_ = value_trainer["optimizer"].asString();
  }

  if (value_trainer.isMember("embedding_optimizer") &&
      value_trainer["embedding_optimizer"].type() == Json::stringValue) {
    if (value_trainer["embedding_optimizer"].asString() != "") {
      optimizer_ = value_trainer["embedding_optimizer"].asString();
    }
  }

  if (value_trainer.isMember("dragon_pipeline_str") &&
      value_trainer["dragon_pipeline_str"].type() == Json::stringValue) {
    dragon_pipeline_str_ = value_trainer["dragon_pipeline_str"].asString();
  }

  if (value_trainer.isMember("btq_topic_num") &&
      value_trainer["btq_topic_num"].type() == Json::intValue) {
    btq_topic_num_ = value_trainer["btq_topic_num"].asInt();
  }

  if (value_trainer.isMember("queue_str_size") &&
      value_trainer["queue_str_size"].type() == Json::intValue) {
    queue_str_size_ = value_trainer["queue_str_size"].asInt();
  }

  if (value_trainer.isMember("dragonfly") &&
      value_trainer["dragonfly"].type() == Json::booleanValue) {
    dragonfly_ = value_trainer["dragonfly"].asBool();
  }

  if (value_trainer.isMember("neg_label") &&
      value_trainer["neg_label"].type() == Json::intValue) {
    neg_label_ = value_trainer["neg_label"].asInt();
  }

  if (value_trainer.isMember("fake_missed_feature") &&
      value_trainer["fake_missed_feature"].type() == Json::booleanValue) {
    fake_missed_feature_ = value_trainer["fake_missed_feature"].asBool();
  }

  if (value_trainer.isMember("batched_sample") &&
      value_trainer["batched_sample"].type() == Json::booleanValue) {
    batched_sample_ = value_trainer["batched_sample"].asBool();
  }

  if (value_trainer.isMember("is_input_one_column") &&
      value_trainer["is_input_one_column"].type() == Json::booleanValue) {
    is_input_one_column_ = value_trainer["is_input_one_column"].asBool();
  }

  if (value_trainer.isMember("kafka_tag") &&
      value_trainer["kafka_tag"].type() == Json::stringValue) {
    if (value_trainer["kafka_tag"].asString() != "") {
      kafka_tag_ = value_trainer["kafka_tag"].asString();
    }
  }

  if (value_trainer.isMember("feed_queue_max_size") &&
      value_trainer["feed_queue_max_size"].type() == Json::intValue) {
    feed_queue_max_size_ = value_trainer["feed_queue_max_size"].asInt();
  }

  if (value_trainer.isMember("shuffle") &&
      value_trainer["shuffle"].type() == Json::booleanValue) {
    shuffle_ = value_trainer["shuffle"].asBool();
  }

  if (value_trainer.isMember("shuffle_size") &&
      value_trainer["shuffle_size"].type() == Json::intValue) {
    shuffle_size_ = value_trainer["shuffle_size"].asInt();
  }

  if (value_trainer.isMember("parallel_eq_cpu_core") &&
      value_trainer["parallel_eq_cpu_core"].type() == Json::booleanValue) {
    parallel_eq_cpu_core_ = value_trainer["parallel_eq_cpu_core"].asBool();
  }

  if (value_trainer.isMember("feed_worker") &&
      value_trainer["feed_worker"].type() == Json::intValue) {
    feed_worker_ = value_trainer["feed_worker"].asInt();
  }

  if (value_trainer.isMember("hub_stream_num") &&
      value_trainer["hub_stream_num"].type() == Json::intValue) {
    hub_stream_num_ = value_trainer["hub_stream_num"].asInt();
  }

  if (value_trainer.isMember("hub_stream_size") &&
      value_trainer["hub_stream_size"].type() == Json::intValue) {
    hub_stream_size_ = value_trainer["hub_stream_size"].asInt();
  }

  if (value_trainer.isMember("hub_train_log_processor_num") &&
      value_trainer["hub_train_log_processor_num"].type() == Json::intValue) {
    hub_train_log_processor_num_ = value_trainer["hub_train_log_processor_num"].asInt();
  }

  if (value_trainer.isMember("hub_train_log_processor_size") &&
      value_trainer["hub_train_log_processor_size"].type() == Json::intValue) {
    hub_train_log_processor_size_ = value_trainer["hub_train_log_processor_size"].asInt();
  }

  if (value_trainer.isMember("hub_feed_node_num") &&
      value_trainer["hub_feed_node_num"].type() == Json::intValue) {
    hub_feed_node_num_ = value_trainer["hub_feed_node_num"].asInt();
  }

  if (value_trainer.isMember("hub_feed_node_size") &&
      value_trainer["hub_feed_node_size"].type() == Json::intValue) {
    hub_feed_node_size_ = value_trainer["hub_feed_node_size"].asInt();
  }

  if (value_trainer.isMember("feature_inclusion_freq") &&
      value_trainer["feature_inclusion_freq"].type() == Json::intValue) {
    feature_inclusion_freq_ = value_trainer["feature_inclusion_freq"].asInt();
  }

  if (value_trainer.isMember("is_kafka_feature") &&
      value_trainer["is_kafka_feature"].type() == Json::booleanValue) {
    is_kafka_feature_ = value_trainer["is_kafka_feature"].asBool();
  }

  if (value_trainer.isMember("dragon_queue_size") &&
      value_trainer["dragon_queue_size"].type() == Json::intValue) {
    dragon_queue_size_ = value_trainer["dragon_queue_size"].asInt();
  }

  if (value_trainer.isMember("use_param_vector") &&
      value_trainer["use_param_vector"].type() == Json::booleanValue) {
    use_param_vector_ = value_trainer["use_param_vector"].asBool();
  }

  if (value_trainer.isMember("use_auto_shard") &&
      value_trainer["use_auto_shard"].type() == Json::booleanValue) {
    use_auto_shard_ = value_trainer["use_auto_shard"].asBool();
  }

  if (value_trainer.isMember("top_ps") &&
      value_trainer["top_ps"].type() == Json::intValue) {
    top_ps_ = value_trainer["top_ps"].asInt();
  }

  if (value_trainer.isMember("top_field") &&
      value_trainer["top_field"].type() == Json::intValue) {
    top_field_ = value_trainer["top_field"].asInt();
  }

  if (value_trainer.isMember("field_shard_limit") &&
      value_trainer["field_shard_limit"].type() == Json::intValue) {
    field_shard_limit_ = value_trainer["field_shard_limit"].asInt();
  }

  if (value_trainer.isMember("update_shard_limit") &&
      value_trainer["update_shard_limit"].type() == Json::intValue) {
    update_shard_limit_ = value_trainer["update_shard_limit"].asInt();
  }

  if (value_trainer.isMember("step_limit") &&
      value_trainer["step_limit"].type() == Json::intValue) {
    step_limit_ = value_trainer["step_limit"].asInt();
  }

  if (value_trainer.isMember("is_move_shard") &&
      value_trainer["is_move_shard"].type() == Json::booleanValue) {
    is_move_shard_ = value_trainer["is_move_shard"].asBool();
  }

  if (value_trainer.isMember("dirname") &&
      value_trainer["dirname"].type() == Json::stringValue) {
    dirname_ = value_trainer["dirname"].asString();
  }

  SPDLOG_INFO("eps[{}]|decay[{}]|l2[{}]|version[{}]|use_freq_scale[{}]", eps_,
              decay_, l2_, version_, use_freq_scale_);

  return true;
}

bool TrainConfig::post_parse() {
  dense_total_size_ = 0;
  for (auto& dense_feature : input_dense_) {
    if (dense_fcs_.find(dense_feature) == dense_fcs_.end()) {
      SPDLOG_ERROR("dense feature[{}] not find config", dense_feature);
      return false;
    }

    dense_total_size_ += dense_fcs_[dense_feature].dim();
    input_dense_dim_.push_back(dense_fcs_[dense_feature].dim());
  }

  sparse_total_size_ = 0;
  for (size_t i = 0; i < input_sparse_.size(); i++) {
    auto sparse_feature = input_sparse_[i];
    auto sparse_iter = sparse_fcs_.find(sparse_feature);
    auto seq_iter = seq_fcs_.find(sparse_feature);
    if (sparse_iter == sparse_fcs_.end() && seq_iter == seq_fcs_.end()) {
      SPDLOG_ERROR("sparse feature[{}] not find config", sparse_feature);
      return false;
    }
    if (sparse_iter != sparse_fcs_.end()) {
      auto columns = sparse_fcs_[sparse_feature];
      for (auto& column : columns) {
        auto emb_table = column.emb_table();
        if (emb_tables_.find(emb_table) == emb_tables_.end()) {
          SPDLOG_ERROR("emb table[{}] not find", emb_table);
          return false;
        }
        if (i != column.sparse_index()) {
          continue;
        }
        sparse_total_size_ += emb_tables_[emb_table].dim();
        input_sparse_dim_.push_back(emb_tables_[emb_table].dim());
      }
    } else if (seq_iter != seq_fcs_.end()) {
      auto emb_table = seq_fcs_[sparse_feature].share_emb_table();
      if (emb_tables_.find(emb_table) == emb_tables_.end()) {
        SPDLOG_ERROR("emb table[{}] not find", emb_table);
        return false;
      }
      sparse_total_size_ += emb_tables_[emb_table].dim();
      input_sparse_dim_.push_back(emb_tables_[emb_table].dim());
    }
  }

  SPDLOG_INFO(
      "Trainer({0}) Train Config summary: batch_size[{1}], "
      "dense_total_size[{2}], "
      "sparse_total_size[{3}], dense_feature_count[{4}], "
      "sparse_feature_count[{5}], hub_eps_size[{6}], ps_eps_size[{7}]",
      trainer_id_, batch_size_, dense_total_size_, sparse_total_size_,
      input_dense_.size(), input_sparse_.size(), hub_eps_.size(),
      ps_eps_.size());

  return true;
}

bool TrainConfig::read_file(const char* path, Json::Value* root) {
  std::ifstream ifs;
  ifs.open(path);

  Json::Reader reader;
  if (!reader.parse(ifs, *root)) {
    SPDLOG_ERROR("parse config failed: path is {}", path);
    return false;
  }
  return true;
}

}  // namespace ops
}  // namespace sniper
