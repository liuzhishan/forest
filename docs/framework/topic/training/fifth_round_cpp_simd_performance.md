# 第五轮排查：C++ SIMD 性能

## 测试 `c++` 中 `simd` 性能

`c++` 中 `simd` 加速 `sum` 实现如下

    #include <immintrin.h>

    void sum_8_f32(float* dst, const float* src) {
      __m256 m_dst = _mm256_loadu_ps(dst);
      __m256 m_src = _mm256_loadu_ps(src);
      __m256 m_val = _mm256_add_ps(m_dst, m_src);
      _mm256_storeu_ps(dst, m_val);
    }

    void sum_f32s(const float* src, float* dst, int32_t n) {
      int32_t c = n / 8;
      int32_t offset = 0;

      for (int32_t i = 0; i < c; ++i) {
        sum_8_f32(dst + offset, src + offset);
        offset += 8;
      }

      while (offset < n) {
        dst[offset] += src[offset];
        ++offset;
      }
    }
    
    
测试代码见: `sniper/trainer/trainer/core/test/test.cc`

结果如下, 普通逻辑就比 `rust` 快了 `4` 倍。

`mm256` 比普通 `sum` 快了约 `1` 倍，`mm512` 比普通 `sum` 快了约 `3` 倍。

`simd` 版本相比 `rust` 版本快了 `10` 倍。
    
    I0925 23:55:40.397699 49243 test.cc:32] run_sum_normal, time spend: 1399 milliseconds, count: 10000000
    I0925 23:55:40.939545 49243 test.cc:43] run_sum_simd_mm256, time spend: 541 milliseconds, count: 10000000
    I0925 23:55:41.313560 49243 test.cc:55] run_sum_simd_mm512, time spend: 374 milliseconds, count: 10000000


## `rust` 编译 `release`

普通 `sum` 版本 `rust` 就比 `c++` 慢了 `4` 倍，有点奇怪，都是最简答的 `inline` 函数。
突然想到 `cargo build` 默认编的是 `debug` 版本，编 `release` 试下。

    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:115 - normal sum time spend: 300 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x4 sum time spend: 82 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x8 sum time spend: 44 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x16 sum time spend: 47 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x32 sum time spend: 45 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x16 sum time spend: 0 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x32 sum time spend: 0 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:174 - run_simd_sum_f32_mm256, use mm256 directly, sum time spend: 100 milliseconds, count: 10000000
    2024-09-26T00:05:16 [INFO] util/tests/tests.rs:190 - run_simd_sum_f32_avx512, user avx512, sum time spend: 95 milliseconds, count: 10000000
    
不同计算逻辑加速比如下

| Method                      | Time (ms) | Speedup |
|-----------------------------|-----------|---------|
| Normal sum                  | 300       | 1.00x   |
| SIMD f32x4 (copy to slice)  | 82        | 3.66x   |
| SIMD f32x8 (copy to slice)  | 44        | 6.82x   |
| SIMD f32x16 (copy to slice) | 47        | 6.38x   |
| SIMD f32x32 (copy to slice) | 45        | 6.67x   |
| SIMD f32x16 (no copy)       | 0         | N/A     |
| SIMD f32x32 (no copy)       | 0         | N/A     |
| SIMD mm256                  | 100       | 3.00x   |
| SIMD AVX512                 | 95        | 3.16x   |

从结果看提高很大，甚至普通版本也比 `c++` 快了 `4` 倍。

并且直接使用 `mm256` 和 `avx512` 比 `Simd` 还要慢一些。避免复制的逻辑依然是最快的。

考虑到 `avx512f` 有些机器上没有，编译后经常会遇到 `Illegal instruction` 报错，并且
速度上没有太大优势，因此不编译这些指令。

### `release` 版本效果

`release` 版本启动有 `Illegal Instruction` 报错，重新在 `ps` 的 `cpu` 机器上编译问题解决。

训练速度能到 `133893 examples/sec` 左右, 比之前快了 `1` 倍。但还是有差距。还是有点奇怪，`sum` 都比 `c++` 快几倍了。

`trainer` 监控耗时如下

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 3694  P50: 42099.884304  P95: 70562.050360  P99: 118674.285714  Max: 806923.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 3188  P50: 42605.000000  P95: 62816.326531  P99: 87356.521739  Max: 159149.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 3694  P50: 18961.937716  P95: 41779.007634  P99: 61700.000000  Max: 770757.000000
    [1,0]<stderr>:2024-09-27 00:44:10,309 - INFO [hooks.py:264 - after_run] - 2024-09-27 00:44:10.309090: step 5000, auc = 0.6720 (130.8 it/sec; 133893.2 examples/sec)
    [1,0]<stderr>:2024-09-27 00:44:10,309 - INFO [hooks.py:232 - after_run] - 2024-09-27 00:44:10.309304: step 5000, xentropy_mean:0 = 0.31240645 (130.8 it/sec; 133890.2 examples/sec)
    [1,0]<stderr>:2024-09-27 00:44:10,309 - INFO [hooks.py:232 - after_run] - 2024-09-27 00:44:10.309356: step 5000, prob_mean:0 = 0.09413883 (130.8 it/sec; 133889.8 examples/sec)
    [1,0]<stderr>:2024-09-27 00:44:10,309 - INFO [hooks.py:232 - after_run] - 2024-09-27 00:44:10.309396: step 5000, real_mean:0 = 0.10058594 (130.8 it/sec; 133889.8 examples/sec)

`ps` 耗时监控如下

    2024-09-27T00:50:33 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 20000, p50: 17418, p95: 26653, p99: 51356, max: 55999
    2024-09-27T00:50:33 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 510000, p50: 0, p95: 6652, p99: 17618, max: 46999
    2024-09-27T00:50:33 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 20000, p50: 0, p95: 943, p99: 1049, max: 9999
    2024-09-27T00:50:33 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 20000, p50: 17319, p95: 26627, p99: 51102, max: 55999

`hub` 耗时监控如下

    2024-09-27T00:48:15 [INFO] sniper/util/src/histogram.rs:659 - HubReadMessage statistics in microseconds, total: 7260000, p50: 0, p95: 1006, p99: 1059, max: 5119107
    2024-09-27T00:48:15 [INFO] sniper/util/src/histogram.rs:659 - HubParseProto statistics in microseconds, total: 7270000, p50: 0, p95: 917, p99: 1036, max: 6000

## `ps` 中 `EmbeddingLookup` 创建 `Vec` 和 `Sum` 耗时


`ps` 创建 `Vec` 和 `Sum` 耗时监控如下, 可以看出，创建 `Vec` 几乎和 `Sum` 耗时差不多。这两个步骤是依次串行执行的,
即先创建 `Vec`，再执行 `Sum`。而整个 `EmbeddingLookup` 耗时的几乎就等于创建 `Vec` 和 `Sum` 的耗时相加。

    2024-09-28T03:13:21 [INFO]sniper/util/src/histogram.rs:665 - PsEmbeddingLookupNewVec statistics in microseconds, total: 170000, p50: 6847, p95: 17241, p99: 18017, max: 31999
    2024-09-28T03:13:21 [INFO]sniper/util/src/histogram.rs:665 - PsEmbeddingLookupSum statistics in microseconds, total: 170000, p50: 714, p95: 12842, p99: 17678, max: 107999
    2024-09-28T03:13:21 [INFO]sniper/util/src/histogram.rs:665 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 170000, p50: 718, p95: 12843, p99: 17681, max: 107999

因此，如果能够避免创建 `Vec`，直接使用 `Sum`，`EmbeddingLookup` 的耗时可以减少一半。

由于涉及到所有权以及并发等问题，目前 `rust` 版本的实现思路和 `c++` 有所区别。

`c++` 中 `EmbeddingLookup` 会并发查询多个 `varname` 对应的 `Embedding` 参数，并进行 `sum` 操作。
但是在开始并发查询之前，会直接将一次请求中所有 `varname` 结果对应的 `tensor` 内存申请好，即创建一个大小
为 `batch_size * var_count * embedding_size` 的 `float` 数组，然后每个子任务根据下标进行访问。
只会有一次 `malloc` 操作。

而在 `safe rust` 中，由于多线程任务中的变量都需要满足所有权的要求, 因此如果要将同一个变量给多个线程访问，
则需要使用 `Arc<Mutex>` 的方式加锁, 无法使用类似 `c++` 的实现方式。这样必然会影响性能。

所以目前的实现思路是，`EmbeddingLookup` 中每个 `varname` 会用一个多线程任务访问，每个任务里创建大小为
`batch_size * embedding_size` 的两层 `float` 数组，然后每个线程访问完自己的 `varname` 后，将结果写入
到 `Vec` 中，最后再将多个线程的结果合并。这种方式会导致 `(batch_size + 1) * var_count` 次 `malloc` 操作，
影响性能。

## 使用 `Arc<SyncUnsafeCell<Vec<f32>>>`

有没有办法采用类似 `c++` 的方式，只创建一次 `Vec`，然后多个线程访问同一个 `Vec`，而不采用 `Arc<Mutex>` 的方式 ？

我们可以使用 `Arc<SyncUnsafeCell<Vec<f32>>>`。

`Arc<SyncUnsafeCell<Vec<f32>>>` 是 `rust` 中的一种类型，它结合了 `Arc` 的线程安全性和 `UnsafeCell` 的内部可变性。

- `Arc` 是 `rust` 中的一个智能指针，它提供了线程安全的引用计数，允许多个线程共享同一个指针。
- `SyncUnsafeCell` 是一个内部可变性包装器，它允许在保证线程安全的前提下，提供对内部值的可变访问。

由于不同的 `varname` 在最终结果中的下标是固定且连续的，因此直接按下标进行访问多个线程之间也不会冲突。因此也可以直接
使用 `SyncUnsafeCell` 来避免加锁。这样的实现方式就和 `c++` 的实现方式一样，只会有一次 `malloc` 操作。

### 奇怪的 `cannot find batch when pull lookup queue` 报错

遇到了个奇怪的问题, 在 `ps` 日志中出现了很多 `cannot find batch when pull lookup queue` 的报错。

    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_66, field: 66, batch_id: 18321147173768650952
    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_26, field: 26, batch_id: 18321147173768650952
    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_19, field: 19, batch_id: 18321147173768650952
    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_44, field: 44, batch_id: 18321147173768650952
    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_22, field: 22, batch_id: 18321147173768650952
    2024-09-28T18:16:43 [INFO] ps/src/embedding.rs:829 - push_lookup_queue, varname: embedding_4, field: 4, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:854 - cannot find batch_id in inner when pull_lookup_queue, field: 0, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:854 - cannot find batch_id in inner when pull_lookup_queue, field: 4, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:729 - cannot find batch_message in lookup_queue, field: 4, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:729 - cannot find batch_message in lookup_queue, field: 0, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:854 - cannot find batch_id in inner when pull_lookup_queue, field: 8, batch_id: 18321147173768650952
    2024-09-28T18:16:53 [ERROR] ps/src/embedding.rs:854 - cannot find batch_id in inner when pull_lookup_queue, field: 15, batch_id: 18321147173768650952

有些 `cannot find batch` 的报错，看时间从 `push lookup queue` 到 `pull lookup queue` 经过了 `9s` 左右。有点问题。

`trainer` 是什么时候收到 `batch_id` 以及什么时候 `push grad` 的?

可能是 `push grad` 队列锁导致的 ?

有时候会报错，有时候不会报错。不稳定复现。

`push grad` 中处理逻辑如下

    // 往队列中推数逻辑
    {
      std::lock_guard<std::mutex> lk(grad_list_mutex_);
      LOG(INFO) << "trainer push grad to message queue, batch_id: " << static_cast<uint64_t>(id)
                << ", grad_list_.size(): " << grad_list_.size();

      grad_list_.push_back(msg);

      LOG(INFO) << "trainer after push grad to message queue, batch_id: " << static_cast<uint64_t>(id)
                << ", grad_list_.size(): " << grad_list_.size();
    }

    LOG(INFO) << "trainer notify push grad thread, batch_id: " << static_cast<uint64_t>(id)
              << ", grad_list_.size(): " << grad_list_.size();
    cond_.notify_one();
    LOG(INFO) << "trainer after notify push grad thread, batch_id: " << static_cast<uint64_t>(id)
              << ", grad_list_.size(): " << grad_list_.size();


    // 从队列中取数逻辑
    while (!exit_flag_) {
      std::unique_lock<std::mutex> lk(grad_list_mutex_);
      LOG(INFO) << "trainer push grad thread, grad_list_.size(): " << grad_list_.size();
      if (!cond_.wait_for(lk, std::chrono::seconds(1),
                          [this] { return !grad_list_.empty(); })) {
        // timeout
        continue;
      }
      auto msg = grad_list_.front();
      grad_list_.pop_front();
      lk.unlock();

      LOG(INFO) << "trainer push grad, batch_id: " << static_cast<uint64_t>(msg->batch_id)
                << ", grad_list_.size(): " << grad_list_.size();
      ...
    }


在 `push grad` 中添加日志结果如下

    [1,0]<stderr>:2024-09-28 18:16:43.861863: I trainer/core/operators/kernels/feed_queue.cc:232] trainer feed queue, batch_id: 18321147173768650952
    [1,0]<stderr>:2024-09-28 18:16:43.967456: I trainer/core/operators/kernels/push_grad_kernels.cc:89] trainer get batch_id from tensorflow, batch_id: 18321147173768650952
    [1,0]<stderr>:2024-09-28 18:16:43.967478: I trainer/core/operators/kernels/push_grad_kernels.cc:102] trainer push grad to message queue, batch_id: 18321147173768650952, grad_list_.size(): 989
    [1,0]<stderr>:2024-09-28 18:16:43.967484: I trainer/core/operators/kernels/push_grad_kernels.cc:105] trainer after push grad to message queue, batch_id: 18321147173768650952, grad_list_.size(): 990
    [1,0]<stderr>:2024-09-28 18:16:43.967489: I trainer/core/operators/kernels/push_grad_kernels.cc:112] trainer notify push grad thread, batch_id: 18321147173768650952, grad_list_.size(): 990
    [1,0]<stderr>:2024-09-28 18:16:43.967493: I trainer/core/operators/kernels/push_grad_kernels.cc:115] trainer after notify push grad thread, batch_id: 18321147173768650952, grad_list_.size(): 990
    [1,0]<stderr>:2024-09-28 18:16:43.967766: I trainer/core/operators/kernels/push_grad_kernels.cc:172] trainer push grad thread, grad_list_.size(): 990
    [1,0]<stderr>:2024-09-28 18:16:53.478982: I trainer/core/operators/kernels/push_grad_kernels.cc:182] trainer push grad, batch_id: 18321147173768650952, grad_list_.size(): 1157
    [1,0]<stderr>:2024-09-28 18:16:53.530248: E ./trainer/core/rpc/grpc/grpc_client.h:91] /sniper.Sniper/PushGrad meets grpc error, error_code: 13, error_message: push grad failed! varnames: embedding_0,embedding_4,embedding_8,embedding_11,embedding_15,embedding_19,embedding_22,embedding_26,embedding_33,embedding_37,embedding_40,embedding_44,embedding_48,embedding_51,embedding_55,embedding_59,embedding_62,emb�push grad failed! varnames: embedding_0,embedding_4,embedding_8,embedding_11,embedding_15,embedding_19,embedding_22,embedding_26,embedding_33,embedding_37,embedding_40,embedding_44,embedding_48,embedding_51,embedding_55,embedding_59,embedding_62,embedding_66,embedding_73, batch_id: 18321147173768650952

从日志中看出，从 `msg` 被 `push` 到 `grad_list_`, 到 `cond_.wait` 之间，经过了 `9s` 左右。而队列
长度从 `989` 增加到 `1157`，即增加了 `168` 个 `msg`。

看起来可能是 `push grad` 处理得太慢了导致的。增加线程数或者核数 ?

#### 修改 `push_thread_count_`

将 `push_thread_count_` 增加到 `10`, 不再报错，`grad_list_` 长度基本在 `1` 左右。

    [1,0]<stderr>:2024-09-28 19:08:31.107230: I trainer/core/operators/kernels/push_grad_kernels.cc:90] trainer get batch_id from tensorflow, batch_id: 6043459736879151418
    [1,0]<stderr>:2024-09-28 19:08:31.107256: I trainer/core/operators/kernels/push_grad_kernels.cc:103] trainer push grad to message queue, batch_id: 6043459736879151418, grad_list_.size(): 0
    [1,0]<stderr>:2024-09-28 19:08:31.107262: I trainer/core/operators/kernels/push_grad_kernels.cc:106] trainer after push grad to message queue, batch_id: 6043459736879151418, grad_list_.size(): 1
    [1,0]<stderr>:2024-09-28 19:08:31.107267: I trainer/core/operators/kernels/push_grad_kernels.cc:113] trainer notify push grad thread, batch_id: 6043459736879151418, grad_list_.size(): 1
    [1,0]<stderr>:2024-09-28 19:08:31.107274: I trainer/core/operators/kernels/push_grad_kernels.cc:116] trainer after notify push grad thread, batch_id: 6043459736879151418, grad_list_.size(): 1
    [1,0]<stderr>:2024-09-28 19:08:31.107291: I trainer/core/operators/kernels/push_grad_kernels.cc:183] trainer push grad, batch_id: 6043459736879151418, grad_list_.size(): 0


#### 增加 `trainer` 核数

将 `trainer` 的核数从 `12` 增加到 `20`, `push_thread_count_` 是 `8`。

`trainer` `grad_list_` 长度监控如下

    [1,0]<stderr>:2024-09-28 20:37:37.968651: I trainer/core/operators/kernels/push_grad_kernels.cc:90] trainer get batch_id from tensorflow, batch_id: 2961414860140179981
    [1,0]<stderr>:2024-09-28 20:37:37.968682: I trainer/core/operators/kernels/push_grad_kernels.cc:103] trainer push grad to message queue, batch_id: 2961414860140179981, grad_list_.size(): 0
    [1,0]<stderr>:2024-09-28 20:37:37.968689: I trainer/core/operators/kernels/push_grad_kernels.cc:106] trainer after push grad to message queue, batch_id: 2961414860140179981, grad_list_.size(): 1
    [1,0]<stderr>:2024-09-28 20:37:37.968694: I trainer/core/operators/kernels/push_grad_kernels.cc:113] trainer notify push grad thread, batch_id: 2961414860140179981, grad_list_.size(): 1
    [1,0]<stderr>:2024-09-28 20:37:37.968700: I trainer/core/operators/kernels/push_grad_kernels.cc:116] trainer after notify push grad thread, batch_id: 2961414860140179981, grad_list_.size(): 1

`trainer` 训练速度如下

    [1,0]<stderr>:2024-09-28 20:31:57,710 - INFO [hooks.py:264 - after_run] - 2024-09-28 20:31:57.710182: step 3100, auc = 0.6702 (135.0 it/sec; 138226.6 examples/sec)
    [1,0]<stderr>:2024-09-28 20:31:57,710 - INFO [hooks.py:232 - after_run] - 2024-09-28 20:31:57.710345: step 3100, xentropy_mean:0 = 0.28454855 (135.0 it/sec; 138237.0 examples/sec)
    [1,0]<stderr>:2024-09-28 20:31:57,710 - INFO [hooks.py:232 - after_run] - 2024-09-28 20:31:57.710399: step 3100, prob_mean:0 = 0.09449448 (135.0 it/sec; 138240.1 examples/sec)
    [1,0]<stderr>:2024-09-28 20:31:57,710 - INFO [hooks.py:232 - after_run] - 2024-09-28 20:31:57.710440: step 3100, real_mean:0 = 0.08984375 (135.0 it/sec; 138243.2 examples/sec)

`trainer` 耗时监控如下

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 3767  P50: 40586.672603  P95: 49140.412585  P99: 49900.745028  Max: 65564.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 3775  P50: 44284.000000  P95: 63949.063232  P99: 68009.000000  Max: 68009.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 3762  P50: 16505.277973  P95: 31455.515588  P99: 41345.000000  Max: 75308.000000

`ps` 耗时监控如下

    2024-09-28T20:40:03 [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookup statistics in microseconds, total: 20000, p50: 15127, p95: 17919, p99: 18167, max: 30000
    2024-09-28T20:40:03 [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupSum statistics in microseconds, total: 530000, p50: 752, p95: 7704, p99: 16920, max: 30000
    2024-09-28T20:40:03 [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 530000, p50: 766, p95: 7721, p99: 16924, max: 30000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupDispatch statistics in microseconds, total: 20000, p50: 0, p95: 1001, p99: 1052, max: 5000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupWaiting statistics in microseconds, total: 20000, p50: 15096, p95: 17910, p99: 18160, max: 30000

可以看出 `ps` 中 `EmbeddingLookup` 耗时 `p95` 从之前的 `26653us` 减少到 `17919us`，减少了 `8734us`。
`trainer` 中 `OpsEmbeddingLookup` 耗时 `p95` 从之前的 `70562us` 减少到 `49140us`，减少了 `21422us`。

但是 `push grad` 耗时从之前的 `13799us` 增加到 `63949us`，增加了 `50150us`。


但是奇怪的是训练速度并没有增加。


## 分析与总结

采用 `release` 编译后，`simd` 的加速效果还是没有达到预期，速度只从 `40000 examples/sec` 增加到 `111131 examples/sec`，
增加了约 `3` 倍。

`c++` 版本

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 5508  P50: 27394.852941  P95: 44173.026316  P99: 56622.222222  Max: 363990.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 5512  P50: 9640.076336  P95: 13799.932461  P99: 19276.981132  Max: 23315.000000
    PsEmbeddingLookup statistics => count: 5684  P50: 6928.621908  P95: 9640.070671  P99: 9881.088339  Max: 19403.000000

`rust` 版本

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 2462  P50: 40720.898258  P95: 49352.566453  P99: 59858.974359  Max: 100417.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 2464  P50: 43928.000000  P95: 65649.350649  P99: 73649.350649  Max: 89166.000000
    sniper/util/src/histogram.rs:661 - PsEmbeddingLookup statistics in microseconds, total: 30000, p50: 15253, p95: 18195, p99: 25443, max: 86001

对比如下

| Operation              | C++ Version (μs) | Rust Version (μs) | Difference (μs)        |
|------------------------|------------------|--------------------|------------------------|
| OpsEmbeddingLookup P50 | 27,394           | 40,720             | +13,326 (49% slower)   |
| OpsEmbeddingLookup P95 | 44,173           | 49,352             | +5,179 (12% slower)    |
| OpsPushGrad P50        | 9,640            | 43,928             | +34,288 (356% slower)  |
| OpsPushGrad P95        | 13,799           | 65,649             | +51,850 (376% slower)  |
| PsEmbeddingLookup P50  | 6,928            | 15,253             | +8,325 (120% slower)   |
| PsEmbeddingLookup P95  | 9,640            | 18,195             | +8,555 (89% slower)    |


并且奇怪的是单纯看 `simd sum` 的性能, `rust` 甚至比 `c++` 还要快 `3` 倍。但是整体的 `EmbeddingLookup` 性能却
比 `c++` 慢了 `1` 倍。

从 `trainer` `OpsEmbeddingLookup` 耗时来看，与 `c++` 版本相比差异并不大。从最初的 `70万us` 减少到目前的 `4万us` 左右。

理论上按 `4万us` 估计，每秒能处理 `25` 个 `batch` 的 `EmbeddingLookup`, 按 `10` 个预取线程计算，
每秒能处理 `250` 个 `batch`。换算成训练速度差不多是 `25万 examples/sec`。 基本和 `c++` 版本持平。

所以目前看来还是其他地方的瓶颈导致训练速度比 `c++` 版本还慢一倍, 比如 `push grad` 慢 `3` 倍, `ReadSample` 慢 `1` 倍。
