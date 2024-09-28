# 第四轮排查：SIMD 加速

## `simd` 加速

突然想到 `c++` 版本中 `ps` `EmbeddingLookup` 与 `PushGrad` 都用了 `simd` 加速来计算 `sum`,  `sum` 是
一个高频操作。利用 `__mm256` 系列指令同时计算 `8` 个 `float` 的运算结果。理论上有 `8` 倍的加速，好
像和速度差异差不多。如果用 `__mm512` 系列则能够同时计算 `16` 个 `float`。

`c++` 中实现如下

    void Sum8FloatValues(float* dst, const float* src) {
      __m256 m_dst = _mm256_loadu_ps(dst);
      __m256 m_src = _mm256_loadu_ps(src);
      __m256 m_val = _mm256_add_ps(m_dst, m_src);
      _mm256_storeu_ps(dst, m_val);
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


参考:
- [Nine Rules for SIMD Acceleration of Your Rust Code (Part 1)](https://towardsdatascience.com/nine-rules-for-simd-acceleration-of-your-rust-code-part-1-c16fe639ce21)
- [Nine Rules for SIMD Acceleration of your Rust Code (Part 2)](https://towardsdatascience.com/nine-rules-for-simd-acceleration-of-your-rust-code-part-2-6a104b3be6f3)

使用 `simd` 需要 `rust nightly`, 按如下步骤安装

    apt-get install libssl-dev

    rustup install nightly
    rustup update nightly
    rustup override set nightly
    
    cargo update

    cargo install cargo-simd-detect --force
    cargo simd-detect
    
    export RUSTFLAGS="-C target-feature=+avx2"
    
    export RUSTFLAGS="-C target-cpu=native -C linker=gcc"
    
注意: 使用 `nightly` 编译会报如下错误，需要设置参数 `export RUSTFLAGS="-C linker=gcc"`

    = note: rust-lld: error: sniper/target/debug/build/tensorflow-sys-a6e7af09a3d8a4cc/out/libtensorflow.so: invalid local symbol '_ZN9grpc_core7ExecCtx9exec_ctx_E' in global part of symbol table
              collect2: error: ld returned 1 exit status
              
或者在 .cargo/config.toml 中设置

    [target.x86_64-unknown-linux-gnu]
    rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+avx512f", "-C", "linker=gcc"]

## 测试 `simd` 性能

对比简单的 `32` 个 `f32` `sum` 性能, `Simd<f32, 32>` 表示 `32` 个 `float`, 再对比 `f32x4`, `f32x8`, `f32x16` 的结果。

具体实现见: `util/tests/tests.rs` 中 `test_simd_sum`。

结果如下，可以看出, 几种不同 `simd` 结构的速度差不多，相比普通的 `sum` 都有 `6` 倍多的加速。

    2024-09-25T21:05:18 [INFO] util/tests/tests.rs:115 - normal sum time spend: 4007 milliseconds, count: 10000000
    2024-09-25T21:05:31 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x4 sum time spend: 12415 milliseconds, count: 10000000
    2024-09-25T21:05:37 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x8 sum time spend: 6764 milliseconds, count: 10000000
    2024-09-25T21:05:41 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x16 sum time spend: 4065 milliseconds, count: 10000000
    2024-09-25T21:05:44 [INFO] util/tests/tests.rs:131 - run_simd_sum_flex, copy to slice, f32x32 sum time spend: 2399 milliseconds, count: 10000000
    2024-09-25T21:05:45 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x16 sum time spend: 691 milliseconds, count: 10000000
    2024-09-25T21:05:45 [INFO] util/tests/tests.rs:156 - run_simd_sum_f32_no_copy, no copy, f32x32 sum time spend: 705 milliseconds, count: 10000000
    2024-09-25T21:05:50 [INFO] util/tests/tests.rs:174 - run_simd_sum_f32_mm256, use mm256 directly, sum time spend: 4533 milliseconds, count: 10000000
    2024-09-25T21:05:55 [INFO] util/tests/tests.rs:190 - run_simd_sum_f32_avx512, user avx512, sum time spend: 4924 milliseconds, count: 10000000
    
不同计算逻辑加速比如下

| Method                  | Time (ms) | Speedup |
|-------------------------|-----------|---------|
| normal sum              |      4007 |   1.00x |
| simd f32x4 sum          |     12415 |   0.32x |
| simd f32x8 sum          |      6764 |   0.59x |
| simd f32x16 sum         |      4065 |   0.99x |
| simd f32x32 sum         |      2399 |   1.67x |
| simd no copy f32x16 sum |       691 |   5.80x |
| simd no copy f32x32 sum |       705 |   5.68x |
| simd mm256              |      4533 |   0.88x |
| simd avx512             |      4924 |   0.81x |


备注:
- normal sum: 普通 sum。
- simd f32x4 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x8 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x16 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd f32x32 sum: 每次 sum 后将 `simd` 中的结果复制到原数组的 `slice` 中。
- simd no copy f32x16 sum: 直接将结果 `sum` 到第一个参数。
- simd no copy f32x32 sum: 直接将结果 `sum` 到第一个参数。
- simd mm256: 直接使用 `__mm256` 指令计算，不使用 `Simd`, 但是会将结果 `load` 到原数组。
- simd avx512: 直接使用 `__avx512` 指令计算，不使用 `Simd`, 但是会将结果 `load` 到原数组。


可以看出，没有复制的计算逻辑是最快的，比普通 `sum` 快了 `6` 倍。其他的 `simd` 中 `N` 越大，则速度越快。而直接使用
指令计算的结果和普通 `sum` 差不多。

`f32x4` 甚至慢了 `3` 倍。

速度只涨到了 `61462 examples/sec`, 有点奇怪，`EmbeddingLookup` 耗时统计和之前比稍微快一点，`p95` 是 `164698`。但是和预期差很多。

`ps` 上耗时监控如下

    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 20000, p50: 115117, p95: 135947, p99: 137798, max: 135000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 510000, p50: 13949, p95: 105329, p99: 131676, max: 140000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 20000, p50: 0, p95: 940, p99: 1042, max: 6000
    2024-09-24T19:53:19 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 20000, p50: 115111, p95: 135946, p99: 137798, max: 135000


### 不使用 `simd` 的耗时监控如下

`trainer` 耗时监控, 速度只有 `31856 examples/sec`, 有点奇怪，之前能到 `40000 examples/sec`。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 945  P50: 311674.082314  P95: 373167.408231  P99: 375977.000000  Max: 375977.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 946  P50: 42036.000000  P95: 51813.725490  P99: 64271.000000  Max: 64271.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 947  P50: 10260.135135  P95: 19540.625000  P99: 21908.125000  Max: 27457.000000
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:264 - after_run] - 2024-09-24 20:25:44.570218: step 1300, auc = 0.6708 (31.1 it/sec; 31856.0 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570601: step 1300, xentropy_mean:0 = 0.27869231 (31.1 it/sec; 31853.9 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570662: step 1300, prob_mean:0 = 0.08517376 (31.1 it/sec; 31853.8 examples/sec)
    [1,0]<stderr>:2024-09-24 20:25:44,570 - INFO [hooks.py:232 - after_run] - 2024-09-24 20:25:44.570710: step 1300, real_mean:0 = 0.08789062 (31.1 it/sec; 31853.7 examples/sec)
    
`ps` 耗时监控, 看起来和 `simd` 版本差不多，感觉是 `simd` 没生效。

    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 115100, p95: 135947, p99: 137800, max: 155999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 220000, p50: 12917, p95: 98373, p99: 130281, max: 155999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 878, p99: 1028, max: 999
    2024-09-24T20:31:30 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 115098, p95: 135947, p99: 137800, max: 155999

`hub` 耗时监控如下

    2024-09-24T20:38:49 [INFO] sniper/util/src/histogram.rs:659 - HubParseProto statistics in microseconds, total: 4900000, p50: 0, p95: 933, p99: 1039, max: 8000
    2024-09-24T20:39:19 [INFO] sniper/util/src/histogram.rs:659 - HubReadMessage statistics in microseconds, total: 4880000, p50: 891, p95: 2978, p99: 6894, max: 1622015


对比 `c++` 版本的 `EmbeddingLookup` `p95` 耗时，还不到 `10000`, 相差了 `10` 倍。


### 不同 `simd` 实现的结果

#### `sum_f32_vectors_simd_flex::<16>`

`sum` 实现如下

    pub fn sum_f32_vectors_simd_flex<const N: usize>(a: &mut Vec<f32>, b: &Vec<f32>)
    where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
        std::simd::Simd<f32, N>: SimdFloat,
    {
        let len = a.len();
        let simd_len = len - (len % N);

        for i in (0..simd_len).step_by(N) {
            let a_chunk = Simd::<f32, N>::from_slice(&a[i..i+N]);
            let b_chunk = Simd::<f32, N>::from_slice(&b[i..i+N]);
            let sum = a_chunk + b_chunk;
            sum.copy_to_slice(&mut a[i..i+N]);
        }

        // Handle remaining elements
        for i in simd_len..len {
            a[i] += b[i];
        }
    }

经检查是参数给错了，`simd` 参数给成了 `4`, 应该给 `16`，和 `embedding_size` 一样，修改后速度能到 `66000 exampls/sec`,
与预期还是有差距。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1916  P50: 139392.971246  P95: 164148.000000  P99: 164148.000000  Max: 164148.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1920  P50: 42103.000000  P95: 49243.059193  P99: 49926.977475  Max: 56925.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1917  P50: 15857.534247  P95: 34858.333333  P99: 47638.333333  Max: 95223.000000
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:264 - after_run] - 2024-09-25 00:55:27.114341: step 4600, auc = 0.6717 (64.2 it/sec; 65703.3 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114695: step 4600, xentropy_mean:0 = 0.27121678 (64.2 it/sec; 65696.7 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114824: step 4600, prob_mean:0 = 0.08899237 (64.2 it/sec; 65694.2 examples/sec)
    [1,0]<stderr>:2024-09-25 00:55:27,114 - INFO [hooks.py:232 - after_run] - 2024-09-25 00:55:27.114867: step 4600, real_mean:0 = 0.08007812 (64.2 it/sec; 65694.1 examples/sec)


`ps` 耗时监控如下

    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 54313, p95: 128356, p99: 136274, max: 104000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 230000, p50: 6843, p95: 51800, p99: 94071, max: 104000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 957, p99: 1044, max: 2000
    2024-09-25T00:57:38 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 54182, p95: 128022, p99: 136208, max: 104000


####  `sum_f32_vectors_simd_no_copy::<16>`

`sum` 实现如下

    pub fn sum_f32_vectors_simd_no_copy<const N: usize>(a: &mut Simd<f32, N>, b: &Vec<f32>)
    where
        std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
        std::simd::Simd<f32, N>: SimdFloat,
    {
        let b_chunk = Simd::<f32, N>::from_slice(b.as_slice());
        *a += b_chunk;
    }


`trainer` 监控耗时如下

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1829  P50: 135684.803002  P95: 159176.000000  P99: 159176.000000  Max: 159176.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1828  P50: 42006.000000  P95: 51919.191919  P99: 61435.000000  Max: 61435.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1829  P50: 13885.297619  P95: 21867.650677  P99: 31383.000000  Max: 115997.000000
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:264 - after_run] - 2024-09-25 21:01:31.881627: step 9900, auc = 0.6665 (61.1 it/sec; 62592.1 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881838: step 9900, xentropy_mean:0 = 0.29910940 (61.1 it/sec; 62591.8 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881893: step 9900, prob_mean:0 = 0.10830545 (61.1 it/sec; 62591.8 examples/sec)
    [1,0]<stderr>:2024-09-25 21:01:31,881 - INFO [hooks.py:232 - after_run] - 2024-09-25 21:01:31.881932: step 9900, real_mean:0 = 0.09472656 (61.1 it/sec; 62591.8 examples/sec)


`ps` 耗时监控如下

    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookup statistics in microseconds, total: 10000, p50: 51209, p95: 60430, p99: 61249, max: 78000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 220000, p50: 3384, p95: 43359, p99: 57832, max: 78000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupDispatch statistics in microseconds, total: 10000, p50: 0, p95: 914, p99: 1035, max: 1000
    2024-09-25T21:02:02 [INFO] sniper/util/src/histogram.rs:659 - PsEmbeddingLookupWaiting statistics in microseconds, total: 10000, p50: 51208, p95: 60428, p99: 61247, max: 78000

`hub` 耗时监控如下

    2024-09-25T21:02:36 [INFO] sniper/util/src/histogram.rs:659 - HubReadMessage statistics in microseconds, total: 4040000, p50: 714, p95: 1064, p99: 1532, max: 3666077
    2024-09-25T21:02:36 [INFO] sniper/util/src/histogram.rs:659 - HubParseProto statistics in microseconds, total: 4050000, p50: 0, p95: 971, p99: 1046, max: 7000

从 `ps` 耗时监控来看，比之前快了一倍，训练速度能到 `6万`。

还是有点奇怪，速度差了很多。待继续排查。

还有个怀疑点: `c++` 中 `EmbeddingLookup` 结果是申请了一个 `var_cont * batch_size * embedding_size` 大小的数组保存结果，
所有变量只申请了一次内存，而 `rust` 中每个变量用了 `Vec<Vec<f32>>` 两层 `Vec` 来保存，因此有 `batch_size + 1` 次申请，
最终结果又需要复制到 `TensorMessage` 中，可能也有影响。

### `EmbeddingLookup` 结果申请一次内存

## 分析与总结

添加了 `simd` 逻辑后速度并没有提高太多, 有点奇怪。并且 `simd` 复制与不复制的逻辑都试过。
`c++` 中也有复制逻辑，即 `sum` 后把 `simd` 的结果 `load` 到内存中，也有点奇怪。性能不应该差这么多。

可能要对比下 `c++` 中的逻辑。
