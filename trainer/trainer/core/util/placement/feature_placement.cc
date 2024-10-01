#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

#include "glog/logging.h"
#include "trainer/core/util/placement/feature_placement.h"

namespace sniper {

uint64_t simple_string_to_int_hash(const std::string& s) {
    uint64_t hash = 0;

    for (char c : s) {
        hash = hash * 31 + static_cast<unsigned char>(c);
    }

    return hash;
}

const std::vector<std::string>& FeaturePlacement::GetEmbPlacement(const std::string& var_name) {
  auto it = placement_.find(var_name);
  if (it != placement_.end()) {
    return it->second;
  }
  LOG(FATAL) << "could not find placement, varname: " << var_name;
}

std::string FeaturePlacement::GetDensePlacement(const std::string& var_name) {
  uint64_t hash = simple_string_to_int_hash(var_name);

  LOG(INFO) << "[FeaturePlacement.GetDensePlacement] var_name: " << var_name
            << ", hash: " << hash;
  return ps_eps_[hash % ps_eps_.size()];
}

void FeaturePlacement::UpdateSparsePlacement(const std::vector<std::vector<size_t>>& new_ps_shard) {
  std::lock_guard<std::mutex> lk(mu_);

  std::unordered_map<std::string, std::vector<std::string>> tmp_placement;

  for (size_t field = 0; field < new_ps_shard.size(); field++) {
    std::string varname = std::string("embedding_") + std::to_string(field);
    for (size_t ps_index : new_ps_shard[field]) {
      if (ps_index < ps_eps_.size()) {
        tmp_placement[varname].push_back(ps_eps_[ps_index]);
      } else {
        LOG(ERROR) << "out of range, ps_index: " << ps_index
                   << ", ps_eps_.size(): " << ps_eps_.size();
      }
    }
  }

  for (auto it = tmp_placement.begin(); it != tmp_placement.end(); it++) {
    placement_[it->first] = it->second;
    ps_shard_[it->first] = it->second;
  }
}

void FeaturePlacement::init() {
  for (auto& ps : ps_eps_) {
    loads_[ps] = 0.0;
  }

  int32_t ps_size = ps_eps_.size();
  while ((ps_size /= 2) > 0) {
    max_shard_ *= 2;
  }

  for (auto it = ps_shard_.begin(); it != ps_shard_.end(); it++) {
    placement_[it->first] = it->second;
    for (auto& ps_name: it->second) {
      loads_[ps_name] += emb_load_.find(it->first)->second / it->second.size();
    }

    std::string join_eps;
    std::for_each(it->second.begin(), it->second.end(),
                  [&](const std::string& piece) { join_eps += (piece + ";"); });
    LOG(INFO) << "prealloc key: " << it->first << ", ps: " << join_eps;
  }

  for (auto& it : loads_) {
    auto& ep = it.first;
    auto& load = it.second;
    LOG(INFO) << "[EmbPlacement] ps loads: ep: " << ep << ", load: " << load;
  }
}

std::vector<std::pair<std::string, float>>
FeaturePlacement::get_sorted_ps_load() {
  std::vector<std::pair<std::string, float>> elems(loads_.begin(),
                                                   loads_.end());
  std::sort(
      elems.begin(), elems.end(),
      [](std::pair<std::string, float> l, std::pair<std::string, float> r) {
        if (l.second < r.second) {
          return true;
        } else if (l.second > r.second) {
          return false;
        } else {
          return l.first < r.first;
        }
      });

  return elems;
}

}  // namespace sniper
