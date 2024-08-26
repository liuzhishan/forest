#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "trainer/core/base/config.h"
#include "trainer/core/base/status.h"
#include "trainer/core/proto/train_config.pb.h"

namespace sniper {

class FeaturePlacement {
 public:
  FeaturePlacement(const std::vector<EmbeddingTable>& vars,
                   const std::vector<std::string>& ps_eps,
                   const std::unordered_map<std::string, std::vector<std::string>>& ps_shard,
                   const std::unordered_map<std::string, float>& emb_load)
    : vars_(vars),
    ps_eps_(ps_eps),
    ps_shard_(ps_shard),
    emb_load_(emb_load) {
    init();
  }
  ~FeaturePlacement() = default;

  // 需要兼顾cpu && 带宽 balance
  const std::vector<std::string>& GetEmbPlacement(const std::string& var_name);
  std::string GetDensePlacement(const std::string& var_name);

  void UpdateSparsePlacement(const std::vector<std::vector<size_t>>& new_ps_shard);

 private:
  void init();
  std::vector<std::pair<std::string, float>> get_sorted_ps_load();

  int32_t calc_shard_num(float load) {
    // TODO(dx) 经验值，需要更好的启发式算法
    if (load >= 40000.0) {
      return std::min(8, max_shard_);
    } else if (load >= 20000.0) {
      return std::min(4, max_shard_);
    } else if (load >= 10000.0) {
      return std::min(2, max_shard_);
    } else {
      return 1;
    }
  }

  std::vector<EmbeddingTable> vars_;
  std::vector<std::string> ps_eps_;
  int32_t max_shard_ = 1;
  std::mutex mu_;

  //var_name(embedding_tab_name) -> ps_nums
  std::unordered_map<std::string, std::vector<std::string>> ps_shard_;
  // var -> shards ep
  std::unordered_map<std::string, std::vector<std::string>> placement_;
  // ep -> load
  std::unordered_map<std::string, float> loads_;
  // emb -> load
  std::unordered_map<std::string, float> emb_load_;

  static FeaturePlacement* _self;

  SNIPER_NOT_COPYABLE_AND_MOVABLE(FeaturePlacement)
};

class FeaturePlacementWrapper {
 public:
  static FeaturePlacement* GetInstance(const std::vector<EmbeddingTable>& vars,
                                       const std::vector<std::string>& ps_eps,
                                       const std::unordered_map<std::string, std::vector<std::string>>& ps_shard,
                                       const std::unordered_map<std::string, float>& emb_load) {
    static FeaturePlacement placement(vars, ps_eps, ps_shard, emb_load);
    return &placement;
  }

 private:
  FeaturePlacementWrapper() = default;
  ~FeaturePlacementWrapper() = default;
};

}  // namespace sniper
