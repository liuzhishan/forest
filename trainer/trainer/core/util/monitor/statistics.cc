#include "trainer/core/util/monitor/statistics.h"

#include <inttypes.h>
#include <math.h>

#include <cassert>
#include <limits>
#include <map>
#include <vector>

namespace sniper {
namespace monitor {

const char *HistogramTypeName(HistogramType type) {
  switch (type) {
    case HistogramType::kPsCreate:
      return "PsCreate";
    case HistogramType::kPsFeedSample:
      return "PsFeedSample";
    case HistogramType::kPsPull:
      return "PsPull";
    case HistogramType::kPsPush:
      return "PsPush";
    case HistogramType::kPsEmbeddingLookup:
      return "PsEmbeddingLookup";
    case HistogramType::kPsPushGrad:
      return "PsPushGrad";
    case HistogramType::kPsSave:
      return "PsSave";
    case HistogramType::kPsRestore:
      return "PsRestore";
    case HistogramType::kPsComplete:
      return "PsComplete";
    case HistogramType::kHubStartSample:
      return "HubStartSample";
    case HistogramType::kHubReadSample:
      return "HubReadSample";
    case HistogramType::kHubNext:
      return "HubNext";
    case HistogramType::kHubProcess:
      return "HubProcess";
    case HistogramType::kHubFeedSample:
      return "HubFeedSample";
    case HistogramType::kHubCountMessageTimeout:
        return "HubCountMessageTimeout";
    case HistogramType::kHubCountMessage:
      return "HubCountMessage";
    case HistogramType::kHubCountAfterTabFilter:
      return "HubCountAfterTabfilter";
    case HistogramType::kHubCountAfterItemFilter:
      return "HubCountAfterItemFilter";
    case HistogramType::kHubCountAfterLabelExtractor:
      return "HubCountAfterLabelExtractor";
    case HistogramType::kHubCountSamplePos:
      return "HubCountSamplePos";
    case HistogramType::kHubCountSampleNeg:
      return "HubCountSampleNeg";
    case HistogramType::kHubKafkaConsume:
      return "HubKafkaConsume";
    case HistogramType::kHubDecompress:
      return "HubDecompress";
    case HistogramType::kHubParseProto:
      return "HubParseProto";
    case HistogramType::kHubKafkaLogProcess:
      return "HubKafkaLogProcess";
    case HistogramType::kHubHandleKafkaFeature:
      return "HubHandleKafkaFeature";
    case HistogramType::kHubDragonFlyRead:
      return "HubDragonFlyRead";
    case HistogramType::kHubDragonFly:
      return "HubDragonFly";
    case HistogramType::kHubSplitBatchedSamples:
      return "HubSplitBatchedSamples";
    case HistogramType::kHubBatchProcessor:
      return "HubBatchProcessor";
    case HistogramType::kHubDragonInputRead:
      return "HubDragonInputRead";
    case HistogramType::kHubGetRuntime:
      return "HubGetRuntime";
    case HistogramType::kHubDragonInputPushBS:
      return "HubDragonInputPushBS";
    case HistogramType::kOpsCreate:
      return "OpsCreate";
    case HistogramType::kOpsFeedSample:
      return "OpsFeedSample";
    case HistogramType::kOpsPull:
      return "OpsPull";
    case HistogramType::kOpsPush:
      return "OpsPush";
    case HistogramType::kOpsEmbeddingLookup:
      return "OpsEmbeddingLookup";
    case HistogramType::kOpsPushGrad:
      return "OpsPushGrad";
    case HistogramType::kOpsSave:
      return "OpsSave";
    case HistogramType::kOpsRestore:
      return "OpsRestore";
    case HistogramType::kOpsComplete:
      return "OpsComplete";
    case HistogramType::kOpsStartSample:
      return "OpsStartSample";
    case HistogramType::kOpsReadSample:
      return "OpsReadSample";
    case HistogramType::kOpsSaveFeatureCount:
      return "OpsSaveFeatureCount";
    case HistogramType::kMergeNext:
      return "MergeNext";
    case HistogramType::kPsFeedCached:
      return "PsFeedCached";
    case HistogramType::kPsLookupCached:
      return "PsLookupCached";
    default:
      return "<unknown>";
  }
}

void Statistics::PushTime(HistogramType type, uint64_t time) {
  histograms_[static_cast<uint32_t>(type)].Add(time);
}

void Statistics::GetData(HistogramType type, HistogramData *data) {
  std::lock_guard<std::mutex> lock(aggregate_lock_);
  histograms_[static_cast<uint32_t>(type)].Data(data);
}

std::string Statistics::ToString(HistogramType type) const {
  std::lock_guard<std::mutex> lock(aggregate_lock_);
  return histograms_[static_cast<uint32_t>(type)].ToString();
}

std::string Statistics::ToString() const {
  std::string result;

  std::lock_guard<std::mutex> lock(aggregate_lock_);
  for (uint32_t i = 0; i < kHistogramTypeNum; ++i) {
    auto &h = histograms_[i];
    if (h.num() == 0) continue;

    char buffer[16384] = {'\0'};
    HistogramData data;
    h.Data(&data);
    snprintf(buffer, 16384,
             "%s statistics => count: %" PRIu64
             "  P50: %f  P95: %f  P99: %f  Max: %f\n",
             HistogramTypeName(static_cast<HistogramType>(i)), h.num(),
             data.median, data.percentile95, data.percentile99, data.max);
    result.append(buffer);
  }

  return result;
}

void Statistics::Reset() {
  std::lock_guard<std::mutex> lock(aggregate_lock_);
  for (auto &h : histograms_) {
    h.Clear();
  }
}

}  // namespace monitor
}  // namespace sniper
