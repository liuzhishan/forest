#include <immintrin.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <chrono>

inline void sum_8_f32_mm256(float* dst, const float* src) {
  __m256 m_dst = _mm256_loadu_ps(dst);
  __m256 m_src = _mm256_loadu_ps(src);
  __m256 m_val = _mm256_add_ps(m_dst, m_src);
  _mm256_storeu_ps(dst, m_val);
}

inline void sum_f32s_mm256(const float* src, float* dst, int32_t n) {
  int32_t c = n / 8;
  int32_t offset = 0;

  for (int32_t i = 0; i < c; ++i) {
    sum_8_f32_mm256(dst + offset, src + offset);
    offset += 8;
  }

  while (offset < n) {
    dst[offset] += src[offset];
    ++offset;
  }
}

inline void sum_8_f32_mm512(float* dst, const float* src) {
  __m512 m_dst = _mm512_loadu_ps(dst);
  __m512 m_src = _mm512_loadu_ps(src);
  __m512 m_val = _mm512_add_ps(m_dst, m_src);
  _mm512_storeu_ps(dst, m_val);
}

inline void sum_f32s_mm512(const float* src, float* dst, int32_t n) {
  int32_t c = n / 16;
  int32_t offset = 0;

  for (int32_t i = 0; i < c; ++i) {
    sum_8_f32_mm512(dst + offset, src + offset);
    offset += 16;
  }

  while (offset < n) {
    dst[offset] += src[offset];
    ++offset;
  }
}

inline void sum_f32s_normal(float* dst, const float* src, size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] += src[i];
  }
}

void run_sum_normal(int32_t n, std::vector<float>& a, const std::vector<float>& b);

void run_sum_simd_mm256(int32_t n, std::vector<float>& a, const std::vector<float>& b);

void run_sum_simd_mm512(int32_t n, std::vector<float>& a, const std::vector<float>& b);

class Timer {
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
  Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

  void reset() {
    start_time = std::chrono::high_resolution_clock::now();
  }

  int64_t elapsed_milliseconds() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  }
};

std::vector<float> gen_random_f32(size_t n);
