#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "feature_placement.h"
#include "trainer/core/base/status.h"
#include "trainer/core/proto/meta.pb.h"
#include "trainer/core/proto/train_config.pb.h"
#include "trainer/core/util/placement/feature_placement.h"

namespace sniper {

// feed queue 请求 embedding look up 耗时。
struct LookupInfo {
  explicit LookupInfo(size_t field, uint64_t time_spend): field(field), time_spend(time_spend) {}

  // 特征 field
  size_t field = 0;
  // 微秒
  uint64_t time_spend = 0;
};

class AutoShard {
 private:
  // 记录请求耗时。
  // 第一层 ps_index, value 是请求耗时的记录。
  std::vector<std::vector<LookupInfo>> lookup_infos_;

  // 最初的 ps_shard, 每个 field 随机分一个 ps
  // field -> [ps_index]
  std::vector<std::vector<size_t>> origin_ps_shard_;
  // field -> [ps_index]
  std::vector<std::vector<size_t>> new_ps_shard_;
  // 新增加的 shard
  // ps -> [field]
  std::vector<std::vector<size_t>> new_alloc_shard_;

  int ps_count_ = 0;
  int sparse_count_ = 0;

  // 取 top ps 进行重新分配
  int top_ps_;
  // top field 个数
  int top_field_;
  // 每个 field 对应的 shard 上限
  int field_shard_limit_;
  // 每个多少步自动计算 shard
  int step_limit_ = 1000;
  int ps_request_count_ = 0;
  int update_shard_limit_ = 0;
  bool is_move_shard_ = true;

  // 更新的次数
  int update_time_ = 0;

  bool is_already_save_ = false;

  // 最多保存多少数据
  const int lookup_info_limit_ = 1000000;

  std::vector<FeaturePlacement*> placements_;

 public:
  static AutoShard& instance() {
    static AutoShard auto_shard;
    return (auto_shard);
  }

  void init(int ps_count,
            const std::vector<std::vector<size_t>>& origin_ps_shard,
            int top_ps = 2,
            int top_field = 2,
            int field_shard_limit = 2,
            int update_shard_limit = 1,
            int step_limit = 1000,
            bool is_move_shard = true);

  template<typename T>
  bool is_in_vector(T x, const std::vector<T>& vec) {
    return std::find(vec.begin(), vec.end(), x) != vec.end();
  }

  bool is_power_of_2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

  void clear();
  void add_lookup_info(size_t ps_index, size_t field, uint64_t time_spend);
  void add_time_spend(const std::unordered_map<std::string, size_t>& ps_to_index,
                      const std::unordered_map<std::string, std::vector<std::string>>& ps_varnames,
                      const std::string& ps_name,
                      const EmbeddingLookupOption& option);

  // 是否已经到达重新计算 shard 的 step
  bool is_ready() ;
  // 新 shard 是否已经稳定
  bool is_finish() const;

  const std::vector<std::vector<size_t>>& new_alloc_shard() const { return (new_alloc_shard_); }

  // 返回 field 对应的 ps 列表。
  std::vector<std::vector<size_t>> compute_shard();
  // 每个 ps 新增的 field 列表。
  std::vector<std::vector<size_t>> compute_new_alloc_shard(
    const std::vector<std::vector<size_t>>& new_ps_shard,
    const std::vector<std::vector<size_t>>& origin_ps_shard);

  void save_new_shard(const std::string& dirname, const std::string& model_name);

  bool is_already_save() const { return is_already_save_; }

  template<typename T>
  bool is_same_vector(const std::vector<T>& a, const std::vector<T>& b);

  const std::vector<std::vector<size_t>>& origin_ps_shard() const { return (origin_ps_shard_); }
  const std::vector<std::vector<size_t>>& new_ps_shard() const { return (new_ps_shard_); }

  int update_time() const { return update_time_; }
  void add_placement(FeaturePlacement* placement) { placements_.push_back(placement); }
  void update_placements();

 private:
  AutoShard() = default;
};

template<typename T>
bool AutoShard::is_same_vector(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) {
    return false;
  }

  size_t n = a.size();
  for (size_t i = 0; i < n; i++) {
    if (i < b.size()) {
      if (a[i] != b[i]) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace sniper
