# 第七轮排查：`PushGrad` 性能

`push grad` 速度主要取决于 `ps` 处理速度。但是 `ps` 的 `EmbeddingLookup` 速度已经
差不多可以和 `c++` 版本持平, 说明查 `map` 与 `sum` 等处理逻辑已经没有明显瓶颈。

还需要对中间步骤进行详细排查。

## 避免创建 `Vec`

`ps` `PushGrad` 中, 与 `EmbeddingLookup` 类似, 会将参数复制到 `Vec` 中, 然后在各线程
子任务重处理。因此也会有创建 `Vec` 太多的开销。而这个参数是只读的，可以考虑采用 `EmbeddingLookup`
优化后的方式处理。

即类似 `ArcUnsafeVec` 的思路，实现 `ArcUnsafeSlice`, 在多线程中直接访问 `slice`, 避免复制 `Vec`。

效果不明显。

`trainer` 速度

    [1,0]<stderr>:2024-09-28 21:42:13,833 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:42:13.833176: step 1300, auc = 0.6660 (135.1 it/sec; 138360.8 examples/sec)
    [1,0]<stderr>:2024-09-28 21:42:14,567 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:42:14.567748: step 1400, auc = 0.6663 (136.1 it/sec; 139401.1 examples/sec)
    [1,0]<stderr>:2024-09-28 21:42:15,309 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:42:15.309360: step 1500, auc = 0.6659 (134.8 it/sec; 138077.4 examples/sec)

`trainer` 耗时监控

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 9  P50: 30250.000000  P95: 35101.000000  P99: 35101.000000  Max: 35101.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 9  P50: 44543.000000  P95: 46831.000000  P99: 46831.000000  Max: 46831.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 9  P50: 4620.000000  P95: 5739.000000  P99: 5739.000000  Max: 5739.000000

`ps` 耗时监控

    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookup statistics in microseconds, total: 30000, p50: 15358, p95: 20810, p99: 26009, max: 27000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupSum statistics in microseconds, total: 670000, p50: 789, p95: 13153, p99: 17409, max: 27000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupOneVariable statistics in microseconds, total: 670000, p50: 800, p95: 13173, p99: 17418, max: 27000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupDispatch statistics in microseconds, total: 30000, p50: 0, p95: 994, p99: 1051, max: 4000
    [INFO] sniper/util/src/histogram.rs:661 - PsEmbeddingLookupWaiting statistics in microseconds, total: 30000, p50: 15293, p95: 19044, p99: 25656, max: 27000


## `ps` 添加 `PushGrad` 详细耗时统计

针对 `PushGrad` 的耗时, 在 `ps` 添加详细的耗时统计。


## 分析与总结

速度能到 `14万 examples/sec`，还是与 `c++` 版本有差距。