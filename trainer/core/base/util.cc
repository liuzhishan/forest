#include <iostream>
#include <sstream>
#include <fstream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <immintrin.h>
#include <net/if.h>
#include <netinet/in.h>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "absl/types/optional.h"
#include "spdlog/spdlog.h"
#include "klearn/core/base/util.h"

namespace klearn {

void Sum8FloatValues(float* dst, const float* src) {
  __m256 m_dst = _mm256_loadu_ps(dst);         // NOLINT
  __m256 m_src = _mm256_loadu_ps(src);         // NOLINT
  __m256 m_val = _mm256_add_ps(m_dst, m_src);  // NOLINT
  _mm256_storeu_ps(dst, m_val);                // NOLINT
}
void Sum(const float* src, float* dst, int32_t n) {
  int32_t c = n / 8;
  int32_t offset = 0;
  for (int32_t i = 0; i < c; ++i) {
    Sum8FloatValues(dst + offset, src + offset);
    offset += 8;
  }
  while (offset < n) {
    dst[offset] += src[offset];
    ++offset;
  }
}
void ApplyAdagrad(const float* w, const float* grad2, const float* grad,
                  size_t len, float eta, float eps, float fgc, float* new_w,
                  float* new_grad2) {
  // NOLINTNEXTLINE
  __m256 c_fgc = _mm256_set_ps(fgc, fgc, fgc, fgc, fgc, fgc, fgc, fgc);
  // NOLINTNEXTLINE
  __m256 c_eta = _mm256_set_ps(eta, eta, eta, eta, eta, eta, eta, eta);
  // NOLINTNEXTLINE
  __m256 c_eps = _mm256_set_ps(eps, eps, eps, eps, eps, eps, eps, eps);
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    // NOLINTNEXTLINE
    __m256 c_grad = _mm256_loadu_ps(grad + i);
    // NOLINTNEXTLINE
    c_grad = _mm256_div_ps(c_grad, c_fgc);
    // NOLINTNEXTLINE
    __m256 cur_acc = _mm256_loadu_ps(grad2 + i);
    // NOLINTNEXTLINE
    __m256 new_acc = _mm256_add_ps(cur_acc, _mm256_mul_ps(c_grad, c_grad));
    // NOLINTNEXTLINE
    __m256 r1 =
        // NOLINTNEXTLINE
        _mm256_div_ps(_mm256_mul_ps(c_grad, c_eta),
                      _mm256_add_ps(c_eps, _mm256_sqrt_ps(new_acc)));
    // NOLINTNEXTLINE
    __m256 c_val = _mm256_loadu_ps(w + i);
    // NOLINTNEXTLINE
    __m256 r = _mm256_sub_ps(c_val, r1);
    // NOLINTNEXTLINE
    _mm256_storeu_ps(new_grad2 + i, new_acc);
    // NOLINTNEXTLINE
    _mm256_storeu_ps(new_w + i, r);
  }
  // int i = 0;
  for (; i < len; ++i) {
    float c_grad = grad[i];
    c_grad /= fgc;
    new_grad2[i] = grad2[i] + c_grad * c_grad;
    new_w[i] = w[i] - c_grad * eta / (eps + sqrt(new_grad2[i]));
  }
}

void ApplyAdagradW(const float* w, const float* grad2, const float* grad,
                   size_t len, float eta, float eps, float decay, float l2,
                   float fgc, float* new_w, float* new_grad2, int version,
                   bool use_freq_scale) {
  // NOLINTNEXTLINE
  __m256 c_fgc = _mm256_set_ps(fgc, fgc, fgc, fgc, fgc, fgc, fgc, fgc);
  // NOLINTNEXTLINE
  __m256 c_eta = _mm256_set_ps(eta, eta, eta, eta, eta, eta, eta, eta);
  // NOLINTNEXTLINE
  __m256 c_eps = _mm256_set_ps(eps, eps, eps, eps, eps, eps, eps, eps);
  // NOLINTNEXTLINE
  __m256 c_decay =
      _mm256_set_ps(decay, decay, decay, decay, decay, decay, decay, decay);
  // NOLINTNEXTLINE
  __m256 c_l2 = _mm256_set_ps(l2, l2, l2, l2, l2, l2, l2, l2);
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    // NOLINTNEXTLINE
    __m256 c_grad = _mm256_loadu_ps(grad + i);
    // NOLINTNEXTLINE
    __m256 c_val = _mm256_loadu_ps(w + i);
    if (use_freq_scale) {
      // NOLINTNEXTLINE
      c_grad = _mm256_add_ps(_mm256_div_ps(c_grad, c_fgc),
                             _mm256_mul_ps(c_val, c_l2));
    } else {
      // NOLINTNEXTLINE
      c_grad = _mm256_add_ps(c_grad, _mm256_mul_ps(c_val, c_l2));
    }
    // NOLINTNEXTLINE
    __m256 cur_acc = _mm256_loadu_ps(grad2 + i);
    // NOLINTNEXTLINE
    __m256 new_acc = _mm256_add_ps(cur_acc, _mm256_mul_ps(c_grad, c_grad));
    __m256 r1;
    if (version == 2) {
      // NOLINTNEXTLINE
      r1 =
          // NOLINTNEXTLINE
          _mm256_add_ps(
              _mm256_mul_ps(c_val, c_decay),
              _mm256_div_ps(_mm256_mul_ps(c_grad, c_eta),
                            _mm256_sqrt_ps(_mm256_add_ps(c_eps, new_acc))));
    } else {
      // NOLINTNEXTLINE
      r1 =
          // NOLINTNEXTLINE
          _mm256_add_ps(
              _mm256_mul_ps(c_val, c_decay),
              _mm256_div_ps(_mm256_mul_ps(c_grad, c_eta),
                            _mm256_add_ps(c_eps, _mm256_sqrt_ps(new_acc))));
    }
    // NOLINTNEXTLINE
    __m256 r = _mm256_sub_ps(c_val, r1);
    // NOLINTNEXTLINE
    _mm256_storeu_ps(new_grad2 + i, new_acc);
    // NOLINTNEXTLINE
    _mm256_storeu_ps(new_w + i, r);
  }
  // int i = 0;
  for (; i < len; ++i) {
    float c_grad = grad[i] + l2 * w[i];
    if (use_freq_scale) {
      c_grad /= fgc;
    }
    new_grad2[i] = grad2[i] + c_grad * c_grad;
    if (version == 2) {
      new_w[i] =
          w[i] - w[i] * decay - c_grad * eta / (sqrt(eps + new_grad2[i]));
    } else {
      new_w[i] =
          w[i] - w[i] * decay - c_grad * eta / (eps + sqrt(new_grad2[i]));
    }
  }
}

void FloatToHalfFloat(const float* source, size_t len, int16_t* target) {
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    __m256 source_val = _mm256_loadu_ps(source + i);                   // NOLINT
    __m128i target_val = _mm256_cvtps_ph(                              // NOLINT
        source_val, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));  // NOLINT
    _mm_storeu_si128((__m128i*)(target + i), target_val);              // NOLINT
  }
  for (; i < len; ++i) {
    target[i] = _cvtss_sh(source[i], 0);  // NOLINT
  }
}

void HalfFloatToFloat(const int16_t* source, size_t len, float* target) {
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    __m128i source_val = _mm_loadu_si128((__m128i*)(source + i));  // NOLINT
    __m256 target_val = _mm256_cvtph_ps(source_val);               // NOLINT
    _mm256_storeu_ps(target + i, target_val);                      // NOLINT
  }
  for (; i < len; ++i) {
    target[i] = _cvtsh_ss(source[i]);  // NOLINT
  }
}

const char KlearnConst::KLEARN_TRAINER[] = "klearn.trainer";
const char KlearnConst::KLEARN_HUB[] = "klearn.hub";
const char KlearnConst::KLEARN_PS[] = "klearn.ps";

// trainer perf
const char KlearnConst::COUNT_BATCH[] = "count_batch";

// ps perf
const char KlearnConst::COUNT_PULL[] = "count_pull";
const char KlearnConst::COUNT_PUSH[] = "count_push";
const char KlearnConst::PULL_SUCCESS[] = "pull_success";
const char KlearnConst::PULL_FAIL[] = "pull_fail";
const char KlearnConst::PUSH_SUCCESS[] = "push_success";
const char KlearnConst::PUSH_FAIL[] = "push_fail";
const char KlearnConst::BUCKET_SIZE[] = "bucket_size";
const char KlearnConst::ID_COUNT[] = "id_count";

// hub perf
const char KlearnConst::COUNT_MESSAGE[] = "count_message";
const char KlearnConst::COUNT_MESSAGE_TIMEOUT[] = "count_message_timeout";
const char KlearnConst::TOTAL[] = "total";
const char KlearnConst::COUNT_AFTER_TAB_FILTER[] = "count_after_tab_filter";
const char KlearnConst::COUNT_AFTER_ITEM_FILTER[] = "count_after_item_filter";
const char KlearnConst::COUNT_AFTER_LABEL_EXTRACTOR[] = "count_after_label_extractor";
const char KlearnConst::COUNT_SAMPLE[] = "count_sample";
const char KlearnConst::COUNT_POSITIVE[] = "count_positive";
const char KlearnConst::COUNT_NEGATIVE[] = "count_negative";

std::string JoinFloat(const float* arr, int n, const std::string& sep) {
  std::ostringstream oss;

  for (int i = 0; i < n; i++) {
    if (i < n - 1) {
      oss << arr[i] << sep;
    } else {
      oss << arr[i];
    }
  }

  return oss.str();
}

bool IsPathExist(const std::string &s) {
  std::ifstream f(s.c_str());
  return f.good();
}

absl::optional<int> find_int_suffix(const std::string& embedding_name) {
  size_t pos = embedding_name.find("_");
  if (pos != std::string::npos) {
    std::string s = embedding_name.substr(pos + 1);
    if (std::all_of(s.begin(), s.end(),
                    [](char c) { return std::isdigit(c); })) {
      return absl::make_optional(std::stoi(s));
    }
  }

  return absl::nullopt;
}

}  // namespace klearn
