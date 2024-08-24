#pragma once

#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "absl/types/optional.h"
#include "absl/strings/str_join.h"

namespace sniper {

void Sum8FloatValues(float* dst, const float* src);
void Sum(const float* src, float* dst, int32_t n);
inline int DurationByMicrosecond(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
void ApplyAdagrad(const float* w, const float* grad2, const float* grad,
                  size_t len, float eta, float eps, float fgc, float* new_w,
                  float* new_grad2);
void ApplyAdagradW(const float* w, const float* grad2, const float* grad,
                   size_t len, float eta, float eps, float decay, float l2,
                   float fgc, float* new_w, float* new_grad2, int version,
                   bool use_freq_scale);

void FloatToHalfFloat(const float* source, size_t len, int16_t* target);
void HalfFloatToFloat(const int16_t* source, size_t len, float* target);

class KlearnConst {
 public:
  static const char KLEARN_TRAINER[];
  static const char KLEARN_HUB[];
  static const char KLEARN_PS[];

  // trainer perf
  static const char COUNT_BATCH[];

  // ps perf
  static const char COUNT_PULL[];
  static const char COUNT_PUSH[];
  static const char PULL_SUCCESS[];
  static const char PULL_FAIL[];
  static const char PUSH_SUCCESS[];
  static const char PUSH_FAIL[];
  static const char BUCKET_SIZE[];
  static const char ID_COUNT[];

  // hub perf
  static const char COUNT_MESSAGE[];
  static const char COUNT_MESSAGE_TIMEOUT[];
  static const char TOTAL[];
  static const char COUNT_AFTER_TAB_FILTER[];
  static const char COUNT_AFTER_ITEM_FILTER[];
  static const char COUNT_AFTER_LABEL_EXTRACTOR[];
  static const char COUNT_SAMPLE[];
  static const char COUNT_POSITIVE[];
  static const char COUNT_NEGATIVE[];
};

std::string JoinFloat(const float* arr, int n, const std::string& sep=",");
bool IsPathExist(const std::string &s);

absl::optional<int> find_int_suffix(const std::string& embedding_name);

template<typename T>
std::string double_list_to_str(const std::vector<std::vector<T>>& vec) {
  std::ostringstream oss;

  for (size_t i = 0; i < vec.size(); i++) {
    oss << "i: " << i << ", value: " << absl::StrJoin(vec[i], ",") << "\n";
  }

  return oss.str();
}

}  // namespace sniper
