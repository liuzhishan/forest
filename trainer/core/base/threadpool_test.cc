#include "core/base/threadpool.h"

#include <gtest/gtest.h>

#include <atomic>

klearn::ThreadPool pool(10);

void do_sum(std::vector<std::future<void>>* fs, std::mutex* mu,
            std::atomic<int>* sum, int cnt) {
  for (int i = 0; i < cnt; ++i) {
    std::lock_guard<std::mutex> l(*mu);
    fs->push_back(pool.Run([sum]() { sum->fetch_add(1); }));
  }
}

TEST(ThreadPool, ConcurrentRun) {
  std::atomic<int> sum(0);
  std::vector<std::thread> threads;
  std::vector<std::future<void>> fs;
  std::mutex fs_mu;
  int n = 50;
  // sum = (n * (n + 1)) / 2
  for (int i = 1; i <= n; ++i) {
    std::thread t(do_sum, &fs, &fs_mu, &sum, i);
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto& t : fs) {
    t.wait();
  }
  EXPECT_EQ(sum, ((n + 1) * n) / 2);
}
