#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace sniper {
namespace monitor {

struct HistogramData {
  double median = 0.0;
  double percentile95 = 0.0;
  double percentile99 = 0.0;
  double average = 0.0;
  double standard_deviation = 0.0;
  // zero-initialize new members since old Statistics::histogramData()
  // implementations won't write them.
  double max = 0.0;
};

struct HistogramStat {
  HistogramStat();
  ~HistogramStat() {}

  HistogramStat(const HistogramStat&) = delete;
  HistogramStat& operator=(const HistogramStat&) = delete;

  void Clear();
  bool Empty() const;
  void Add(uint64_t value);
  void Merge(const HistogramStat& other);

  inline uint64_t min() const { return min_.load(std::memory_order_relaxed); }
  inline uint64_t max() const { return max_.load(std::memory_order_relaxed); }
  inline uint64_t num() const { return num_.load(std::memory_order_relaxed); }
  inline uint64_t sum() const { return sum_.load(std::memory_order_relaxed); }
  inline uint64_t sum_squares() const {
    return sum_squares_.load(std::memory_order_relaxed);
  }
  inline uint64_t bucket_at(size_t b) const {
    return buckets_[b].load(std::memory_order_relaxed);
  }

  double Median() const;
  double Percentile(double p) const;
  double Average() const;
  double StandardDeviation() const;
  void Data(HistogramData* const data) const;
  std::string ToString() const;

  // To be able to use HistogramStat as thread local variable, it
  // cannot have dynamic allocated member. That's why we're
  // using manually values from BucketMapper
  std::atomic_uint_fast64_t min_;
  std::atomic_uint_fast64_t max_;
  std::atomic_uint_fast64_t num_;
  std::atomic_uint_fast64_t sum_;
  std::atomic_uint_fast64_t sum_squares_;
  std::atomic_uint_fast64_t buckets_[109];  // 109==BucketMapper::BucketCount()
  const uint64_t num_buckets_;
};

class Histogram {
 public:
  Histogram() { Clear(); }

  Histogram(const Histogram&) = delete;
  Histogram& operator=(const Histogram&) = delete;

  void Clear();
  bool Empty() const;
  void Add(uint64_t value);
  void Merge(const Histogram& other);

  std::string ToString() const;
  const char* Name() const { return "Histogram"; }
  uint64_t min() const { return stats_.min(); }
  uint64_t max() const { return stats_.max(); }
  uint64_t num() const { return stats_.num(); }
  double Median() const;
  double Percentile(double p) const;
  double Average() const;
  double StandardDeviation() const;
  void Data(HistogramData* const data) const;

 private:
  HistogramStat stats_;
  std::mutex mutex_;
};

}  // namespace monitor
}  // namespace sniper
