#include <immintrin.h>
#include <glog/logging.h>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "test.h"

std::vector<float> gen_random_f32(size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  std::vector<float> random_floats;
  for (int i = 0; i < n; ++i) {
      random_floats.push_back(dis(gen));
  }

  return random_floats;
}

void run_sum_normal(int32_t n, std::vector<float>& a, const std::vector<float>& b) {
  Timer timer;

  for (int32_t i = 0; i < n; i++) {
    sum_f32s_normal(a.data(), b.data(), a.size());
  }

  LOG(INFO) << "run_sum_normal, time spend: " << timer.elapsed_milliseconds() << " milliseconds"
            << ", count: " << n;
}

void run_sum_simd_mm256(int32_t n, std::vector<float>& a, const std::vector<float>& b) {
  Timer timer;

  for (int32_t i = 0; i < n; i++) {
    sum_f32s_mm256(b.data(), a.data(), a.size());
  }

  LOG(INFO) << "run_sum_simd_mm256, time spend: " << timer.elapsed_milliseconds()
            << " milliseconds"
            << ", count: " << n;
}

void run_sum_simd_mm512(int32_t n, std::vector<float>& a, const std::vector<float>& b) {
  Timer timer;

  for (int32_t i = 0; i < n; i++) {
    sum_f32s_mm512(b.data(), a.data(), a.size());
  }

  LOG(INFO) << "run_sum_simd_mm512, time spend: " << timer.elapsed_milliseconds()
            << " milliseconds"
            << ", count: " << n;
}

void run_sum_test() {
  std::vector<float> a = gen_random_f32(32);
  std::vector<float> b = gen_random_f32(32);

  const int32_t n = 10000000;

  run_sum_normal(n, a, b);
  run_sum_simd_mm256(n, a, b);
  run_sum_simd_mm512(n, a, b);
}

int main(int argc, const char * argv[]) {
  google::InitGoogleLogging(argv[0]);

  FLAGS_alsologtostderr = 1;

  run_sum_test();

  return 0;
}
