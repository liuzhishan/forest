#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include "absl/types/optional.h"
#include "absl/strings/str_join.h"
#include "auto_shard.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/base/util.h"
#include "glog/logging.h"
#include "include/json/json.h"
#include "trainer/core/util/placement/auto_shard.h"

namespace sniper {

void AutoShard::init(int ps_count,
                     const std::vector<std::vector<size_t>>& origin_ps_shard,
                     int top_ps,
                     int top_field,
                     int field_shard_limit,
                     int update_shard_limit,
                     int step_limit,
                     bool is_move_shard) {
  ps_count_ = ps_count;
  origin_ps_shard_ = origin_ps_shard;
  new_ps_shard_ = origin_ps_shard;
  sparse_count_ = origin_ps_shard.size();

  update_shard_limit_ = update_shard_limit;
  step_limit_ = step_limit;

  top_ps_ = std::min(top_ps, static_cast<int>(ps_count / 2));
  top_field_ = std::min(top_field, sparse_count_);
  field_shard_limit_ = field_shard_limit;
  is_move_shard_ = is_move_shard;

  lookup_infos_.resize(ps_count_);
}

void AutoShard::clear() {
  for (size_t i = 0; i < lookup_infos_.size(); i++) {
    lookup_infos_[i].clear();
  }

  ps_request_count_ = 0;
}

void AutoShard::add_lookup_info(size_t ps_index, size_t field, uint64_t time_spend) {
  if (ps_index < lookup_infos_.size()) {
    if (lookup_infos_[ps_index].size() > lookup_info_limit_) {
      return;
    }

    lookup_infos_[ps_index].emplace_back(field, time_spend);
  } else {
    LOG(INFO) << "out of range, ps_index: " << ps_index
              << ", lookup_infos_.size(): " << lookup_infos_.size();
  }
}

void AutoShard::add_time_spend(const std::unordered_map<std::string, size_t>& ps_to_index,
                               const std::unordered_map<std::string, std::vector<std::string>>& ps_varnames,
                               const std::string& ps_name,
                               const EmbeddingLookupOption& option) {
  ps_request_count_ += 1;

  auto it_ps_index = ps_to_index.find(ps_name);
  if (it_ps_index != ps_to_index.end()) {
    size_t ps_index = it_ps_index->second;
    auto it_varnames = ps_varnames.find(ps_name);
    if (it_varnames != ps_varnames.end()) {
      const auto& varnames = it_varnames->second;
      if (varnames.size() == static_cast<size_t>(option.time_spends_size())) {
        for (size_t idx = 0; idx < option.time_spends_size(); idx++) {
          if (absl::optional<int> field = find_int_suffix(varnames[idx])) {
            add_lookup_info(ps_index, *field, option.time_spends(idx));
          }
        }
      } else {
        LOG(INFO) << "varnames.size() != option.time_spends_size(), varnames.size(): "
                  << varnames.size()
                  << ", option.time_spends_size(): " << option.time_spends_size();
      }
    } else {
      LOG(INFO) << "cannot find varnames, ps_name: " << ps_name;
    }
  } else {
    LOG(INFO) << "cannot find ps_index, ps_name: " << ps_name;
  }
}

bool AutoShard::is_ready() {
  if (ps_request_count_ > 0 && ps_request_count_ >= (step_limit_ * ps_count_)) {
    LOG(INFO) << "auto shard is_ready, ps_request_count_: " << ps_request_count_
              << ", ps_count_: " << ps_count_
              << ", update_time_: " << update_time_;

    return true;
  }

  return false;
}

bool AutoShard::is_finish() const {
  return update_time_ >= update_shard_limit_;
}

std::vector<std::vector<size_t>> AutoShard::compute_shard() {
  // 每个 ps 的 load
  std::vector<uint64_t> ps_load(ps_count_, 0);

  // 每个 ps 上 field 对应的 load
  std::vector<std::vector<uint64_t>> ps_field_load(ps_count_);
  for (size_t i = 0; i < ps_count_; i++) {
    ps_field_load[i].resize(sparse_count_, 0);
  }

  for (size_t ps_index = 0; ps_index < ps_count_; ps_index++) {
    std::vector<int> total_each_field(sparse_count_);
    for (size_t k = 0; k < total_each_field.size(); k++) {
      total_each_field[k] = 0;
    }

    for (size_t j = 0; j < lookup_infos_[ps_index].size(); j++) {
      if (ps_index < ps_field_load.size()) {
        size_t field = lookup_infos_[ps_index][j].field;

        ps_load[ps_index] = std::max(ps_load[ps_index], lookup_infos_[ps_index][j].time_spend);

        if (field < ps_field_load[ps_index].size()) {
          total_each_field[field] += 1;

          ps_field_load[ps_index][field] += lookup_infos_[ps_index][j].time_spend;
        } else {
          LOG(INFO) << "out of range, field: " << field
                    << ", ps_index: " << ps_field_load[ps_index].size()
                    << ", ps_field_load[ps_index].size(): " << ps_field_load[ps_index].size();
        }
      }
    }

    for (size_t field = 0; field < ps_field_load[ps_index].size(); field++) {
      if (field < total_each_field.size()) {
        if (total_each_field[field] > 0) {
          ps_field_load[ps_index][field] = static_cast<uint64_t>(
              ps_field_load[ps_index][field] / total_each_field[field]);
        }
      }
    }

    LOG(INFO) << "ps_index: " << ps_index
              << ", ps_load: " << ps_load[ps_index]
              << ", total_each_field: " << absl::StrJoin(total_each_field, ",")
              << ", ps_field_load: " << absl::StrJoin(ps_field_load[ps_index], ",");
  }

  std::vector<std::pair<size_t, uint64_t>> ps_load_pair;
  for (size_t i = 0; i < ps_load.size(); i++) {
    ps_load_pair.emplace_back(i, ps_load[i]);
  }

  // 降序排，第一个最大
  std::sort(ps_load_pair.begin(),
            ps_load_pair.end(),
            [](const std::pair<size_t, uint64_t>& a, const std::pair<size_t, uint64_t>& b) { 
                return a.second > b.second; 
             });

  origin_ps_shard_ = new_ps_shard_;

  for (size_t i = 0; i < top_ps_; i++) {
    std::vector<std::pair<size_t, uint64_t>> field_load_pair;
    for (size_t j = 0; j < ps_field_load[i].size(); j++) {
      field_load_pair.emplace_back(j, ps_field_load[i][j]);
    }

    // 降序排，第一个最大
    std::sort(field_load_pair.begin(),
              field_load_pair.end(),
              [](const std::pair<size_t, uint64_t>& a, const std::pair<size_t, uint64_t>& b) { 
                return a.second > b.second; 
             });

    for (size_t j = 0; j < top_field_; j++) {
      if (ps_count_ - 1 - i < ps_load_pair.size()) {
        size_t target_field = field_load_pair[j].first;

        if (new_ps_shard_[target_field].size() >= field_shard_limit_) {
          continue;
        }

        absl::optional<size_t> target_ps;
        for (int k = ps_count_ - 1 - i; k >= 0; k--) {
          if (!is_in_vector(ps_load_pair[k].first, new_ps_shard_[target_field])) {
            target_ps.emplace(ps_load_pair[k].first);
            break;
          }
        }

        if (target_ps) {
          if (is_move_shard_) {
            new_ps_shard_[target_field] = {*target_ps};
          } else {
            new_ps_shard_[target_field].push_back(*target_ps);
          }
          LOG(INFO) << "add ps, target_field: " << target_field
                    << ", ps: " << *target_ps;
        }
      }
    }
  }

  for (size_t field = 0; field < new_ps_shard_.size(); field++) {
    // 每个 field 对应的分片数必须是 2 的幂
    if (!is_power_of_2(new_ps_shard_[field].size())) {
      for (size_t j = 0; j < ps_count_; j++) {
        if (!is_in_vector(j, new_ps_shard_[field])) {
          new_ps_shard_[field].push_back(j);
          if (is_power_of_2(new_ps_shard_[field].size())) {
            break;
          }
        }
      }
    }
  }

  new_alloc_shard_ = compute_new_alloc_shard(new_ps_shard_, origin_ps_shard_);

  update_placements();

  return new_ps_shard_;
}

void AutoShard::update_placements() {
  // 更新 placement
  for (size_t i = 0; i < placements_.size(); i++) {
    if (placements_[i] != nullptr) {
      placements_[i]->UpdateSparsePlacement(new_ps_shard_);
    } else {
      LOG(INFO) << "something is wrong, placement is nullptr! i: " <<  i;
    }
  }

  // 必须在更新 placement 之后
  update_time_ += 1;
}

std::vector<std::vector<size_t>> AutoShard::compute_new_alloc_shard(
  const std::vector<std::vector<size_t>>& new_ps_shard,
  const std::vector<std::vector<size_t>>& origin_ps_shard) {

  // field -> [ps_index]
  std::vector<std::vector<size_t>> res(sparse_count_);

  // i 是 field
  // 已经存在的也要重新创建
  for (size_t i = 0; i < new_ps_shard.size(); i++) {
    if (i < origin_ps_shard.size()) {
      // 等于 0 的是序列特征
      if (new_ps_shard[i].size() > 0 && origin_ps_shard[i].size() > 0) {
        if (!is_same_vector(new_ps_shard[i], origin_ps_shard[i])) {
          for (size_t ps_index : new_ps_shard[i]) {
            if (i < res.size()) {
              res[i].push_back(ps_index);
            }
          }
        }
      }
    } else {
      LOG(INFO) << "out of range, i: " << i
                << ", origin_ps_shard.size(): " << origin_ps_shard.size();
    }
  }

  for (size_t i = 0; i < res.size(); i++) {
    if (res[i].size() > 0) {
      LOG(INFO) << "new_alloc_shard, field: " << i
                << ", ps: " << absl::StrJoin(res[i], ",");
    }
  }

  return res;
}

void AutoShard::save_new_shard(const std::string& dirname, const std::string& model_name) {
  if (is_already_save_) {
    return;
  }

  Json::Value res(Json::ValueType::objectValue);

  for (int field = 0; field < new_ps_shard_.size(); field++) {
    Json::Value v(Json::ValueType::arrayValue);
    std::string key = std::to_string(field);
    res[key] = v;
    for (size_t i = 0; i < new_ps_shard_[field].size(); i++) {
      res[key].append(static_cast<int>(new_ps_shard_[field][i]));
    }
  }

  std::string filename = dirname + "/" + model_name + "_auto_shard.json";

  // backup
  std::ifstream ifs(filename);
  if (ifs.is_open()) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d%m%Y%H%M%S");
    std::string backup_filename = dirname + "/" + model_name + "_auto_shard_" + oss.str() + ".json";
    std::ofstream ofs_backup(backup_filename);
    if (ofs_backup.is_open()) {
      ofs_backup << ifs.rdbuf();
      ofs_backup.close();
      LOG(INFO) << "find auto_shard json, save to backup_filename: " << backup_filename;
    }
    ifs.close();
  }

  std::ofstream ofs(filename);
  if (ofs.is_open()) {
    ofs << res;
    ofs.close();

    LOG(INFO) << "save auto_shard to file: " << filename;
  }

  is_already_save_ = true;
}

}  // namespace sniper
