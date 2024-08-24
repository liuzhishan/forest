#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "core/util/monitor/histogram.h"

namespace sniper {
namespace monitor {

enum HistogramType {
  // ps
  kPsCreate = 0,
  kPsFeedSample,
  kPsPull,
  kPsPush,
  kPsEmbeddingLookup,
  kPsPushGrad,
  kPsSave,
  kPsRestore,
  kPsComplete,
  kPsFeedCached,
  kPsLookupCached,

  // hub
  kHubStartSample,
  kHubReadSample,
  kHubNext,
  kHubProcess,
  kHubFeedSample,
  kHubCountMessageTimeout,
  kHubCountMessage,
  kHubCountAfterTabFilter,
  kHubCountAfterItemFilter,
  kHubCountAfterLabelExtractor,
  kHubCountSamplePos,
  kHubCountSampleNeg,
  kHubKafkaConsume,
  kHubDecompress,
  kHubParseProto,
  kHubKafkaLogProcess,
  kHubHandleKafkaFeature,
  kHubDragonFlyRead,
  kHubDragonFly,
  kHubSplitBatchedSamples,
  kHubBatchProcessor,
  kHubDragonInputRead,
  kHubGetRuntime,
  kHubDragonInputPushBS,

  // trainer op
  kOpsCreate,
  kOpsFeedSample,
  kOpsPull,
  kOpsPush,
  kOpsEmbeddingLookup,
  kOpsPushGrad,
  kOpsSave,
  kOpsRestore,
  kOpsComplete,
  kOpsStartSample,
  kOpsReadSample,
  kOpsSaveFeatureCount,
  kOpsRestoreFeatureCount,

  kMax,

  kMergeNext,
};

const char *HistogramTypeName(HistogramType type);

static constexpr uint32_t kHistogramTypeNum =
    static_cast<uint32_t>(HistogramType::kMax);

class Statistics {
 public:
  void PushTime(HistogramType type, uint64_t time);

  void GetData(HistogramType type, HistogramData *data);

  std::string ToString(HistogramType type) const;
  std::string ToString() const;

  void Reset();

 private:
  Histogram histograms_[kHistogramTypeNum];
  mutable std::mutex aggregate_lock_;
};

}  // namespace monitor
}  // namespace sniper
