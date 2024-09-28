# 第六轮排查：`ReadSample` 性能

`ReadSample` 主要依赖于 `hub` 读取数据的速度，比较简单的方法是直接增加 `hub` 个数。

## 增加 `hub` 个数从 `4` 到 `6` 个

区别不大。

速度

    [1,0]<stderr>:2024-09-28 21:00:34,837 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:00:34.837934: step 25000, auc = 0.6596 (135.6 it/sec; 138853.9 examples/sec)
    [1,0]<stderr>:2024-09-28 21:00:35,604 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:00:35.604699: step 25100, auc = 0.6595 (130.4 it/sec; 133547.9 examples/sec)
    [1,0]<stderr>:2024-09-28 21:00:36,350 - INFO [hooks.py:264 - after_run] - 2024-09-28 21:00:36.350687: step 25200, auc = 0.6595 (134.1 it/sec; 137267.7 examples/sec)

`trainer` 耗时监控

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 355  P50: 34118.421053  P95: 48369.000000  P99: 48369.000000  Max: 48369.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 358  P50: 43145.000000  P95: 49856.268222  P99: 59161.000000  Max: 59161.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 352  P50: 7929.496403  P95: 13967.631579  P99: 20740.000000  Max: 22258.000000

## 分析与总结