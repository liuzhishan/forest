# 第三轮排查：feed_queue 耗时

## `feed_queue` 中各步骤的耗时

将请求 `hub`, `ReadSample`, `EmbeddingLookup`, `push to queue` 等步骤都加上日志, `thread_id 0` 结果如下, 可以
看出，大部分 `ReadSample` 请求都没有获取到数据。而如果从 `hub` 获取到数据，后面的 `EmbeddingLookup` 耗时与监控一致。


说明瓶颈是 `hub` ?


    [1,0]<stderr>:2024-09-20 13:48:34.646883: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.646898: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.650838: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.650848: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.654586: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.654607: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.658092: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.658129: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.661794: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.661806: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.665330: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.665343: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.669174: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.669191: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.672761: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.672774: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.676267: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.676285: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.679938: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.679953: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.683477: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.683485: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.687396: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.687403: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.690899: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.690909: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.694596: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.694602: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.698400: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.698407: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.701864: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.701871: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.705676: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.705685: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.709361: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.709374: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.713283: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.713297: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.716862: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.716869: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.726340: I trainer/core/operators/kernels/feed_queue.cc:258] trainer thread id: 0, after read sample, start check dim
    [1,0]<stderr>:2024-09-20 13:48:34.726352: I trainer/core/operators/kernels/feed_queue.cc:277] trainer thread id: 0, after check dim, start embedding lookup
    [1,0]<stderr>:2024-09-20 13:48:34.887918: I trainer/core/operators/kernels/feed_queue.cc:451] trainer thread id: 0, after embedding lookup, start check embedding result
    [1,0]<stderr>:2024-09-20 13:48:34.887940: I trainer/core/operators/kernels/feed_queue.cc:468] trainer thread id: 0, after check embedding result, start push to queue
    [1,0]<stderr>:2024-09-20 13:48:34.898154: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-20 13:48:34.898168: I trainer/core/operators/kernels/feed_queue.cc:151] trainer thread id: 0, start get next_hub
    [1,0]<stderr>:2024-09-20 13:48:34.898173: I trainer/core/operators/kernels/feed_queue.cc:202] trainer thread id: 0, start read sample
    [1,0]<stderr>:2024-09-20 13:48:34.907967: I trainer/core/operators/kernels/feed_queue.cc:258] trainer thread id: 0, after read sample, start check dim
    [1,0]<stderr>:2024-09-20 13:48:34.907981: I trainer/core/operators/kernels/feed_queue.cc:277] trainer thread id: 0, after check dim, start embedding lookup
    [1,0]<stderr>:2024-09-20 13:48:35.106380: I trainer/core/operators/kernels/feed_queue.cc:451] trainer thread id: 0, after embedding lookup, start check embedding result
    [1,0]<stderr>:2024-09-20 13:48:35.106403: I trainer/core/operators/kernels/feed_queue.cc:468] trainer thread id: 0, after check embedding result, start push to queue


## 增加 `hub` 个数

从 `2` 增加到 `4`。

有一点上涨，能到 `38400 exampls/sec`, 但是与预期还是有差距。

    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:269 - after_run] - 2024-09-21 09:21:46.800022: step 1300, auc = 0.6715 (37.5 it/sec; 38399.8 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800209: step 1300, xentropy_mean:0 = 0.30443680 (37.5 it/sec; 38400.1 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800266: step 1300, prob_mean:0 = 0.08712210 (37.5 it/sec; 38400.1 examples/sec)
    [1,0]<stderr>:2024-09-21 09:21:46,800 - INFO [hooks.py:237 - after_run] - 2024-09-21 09:21:46.800310: step 1300, real_mean:0 = 0.09960938 (37.5 it/sec; 38400.1 examples/sec)
    

查看 `after push to queue` 日志结果如下:

    # grep '09:22:27' log/dsp_ctr_lzs_test_v6_2024_09_21_09_19_17.log | grep 'after push to queue'

    [1,0]<stderr>:2024-09-21 09:22:27.053202: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.064763: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.076186: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.080873: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.081193: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.088784: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.102404: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.165555: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.166353: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.200455: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.305205: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.310202: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.333078: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.339847: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.342972: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.363182: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.373962: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.384125: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.431764: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.485752: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.565343: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.572960: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.607563: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.608794: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.611064: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.615670: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.633463: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.638674: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.705027: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.741057: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 7, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.855039: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 4, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.882023: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 5, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.884056: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 8, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.888113: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 6, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.888182: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 2, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.898029: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 9, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.898356: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 0, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.937386: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 3, after push to queue
    [1,0]<stderr>:2024-09-21 09:22:27.962224: I trainer/core/operators/kernels/feed_queue.cc:489] trainer thread id: 1, after push to queue


    grep '09:22:27' log/dsp_ctr_lzs_test_v6_2024_09_21_09_19_17.log | grep 'after push to queue' | wc -l
    
    39
    
去掉 `trainer` 日志，能到 `49772 examples/sec`。

    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1440  P50: 209488.636364  P95: 246306.818182  P99: 249579.545455  Max: 264189.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1440  P50: 42283.000000  P95: 49962.800875  P99: 70073.529412  Max: 91791.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1438  P50: 13212.349914  P95: 25916.822430  P99: 31830.093458  Max: 39568.000000

    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:264 - after_run] - 2024-09-21 14:08:46.568055: step 3600, auc = 0.6713 (48.6 it/sec; 49772.2 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568233: step 3600, xentropy_mean:0 = 0.29283547 (48.6 it/sec; 49772.1 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568289: step 3600, prob_mean:0 = 0.08826549 (48.6 it/sec; 49772.0 examples/sec)
    [1,0]<stderr>:2024-09-21 14:08:46,568 - INFO [hooks.py:232 - after_run] - 2024-09-21 14:08:46.568333: step 3600, real_mean:0 = 0.08984375 (48.6 it/sec; 49772.0 examples/sec)


## 增加 `trainer` 预取线程数

将 `trainer` 预取线程数从 `10` 增加到 `20`, 速度和之前差不多, 离预期还有距离。

    [1,0]<stderr>:2024-09-21 09:29:01,576 - INFO [hooks.py:269 - after_run] - 2024-09-21 09:29:01.576279: step 15700, auc = 0.6591 (37.0 it/sec; 37902.6 examples/sec)
    
耗时监控如下, 可以看出, `EmbeddingLookup` 耗时增加约 `1` 倍, `PushGrad` 基本不变, `ReadSample` `p50` 基本不变，但是 `p95` 增加约 `1` 倍。

    [1,0]<stderr>:I0921 09:38:38.752875   786 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1316  P50: 462387.931034  P95: 559386.206897  P99: 568008.275862  Max: 576471.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1318  P50: 41835.173502  P95: 49786.829653  P99: 68785.714286  Max: 75602.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1316  P50: 15726.872247  P95: 44936.956522  P99: 49800.434783  Max: 92753.000000
    [1,0]<stderr>:I0921 09:39:08.753228   786 run_status.cc:71] [RunStatus]: OpsEmbeddingLookup statistics => count: 1313  P50: 462747.205503  P95: 559274.720550  P99: 567854.944110  Max: 568404.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1308  P50: 41818.000000  P95: 49607.075472  P99: 65916.666667  Max: 68691.000000
    [1,0]<stderr>:OpsReadSample statistics => count: 1310  P50: 13814.439655  P95: 31207.865169  P99: 41405.555556  Max: 99806.000000

资源利用率:

- `hub`: cpu 利用率约 `30%`。
- `ps`: `ps0` 利用率约为 `60%`,  不均匀，其他由于监控问题看不到。

## 经过第二轮排查与优化后与 `c++` 版本对比

`c++` 版本速度能到 `240000 examples/sec`, `rust` 版本约为 `40000 examples/sec`, 相差 `6` 倍。

对比 `rust` `38000 examples/sec` 版本统计耗时

    [1,0]<stderr>:OpsReadSample statistics => count: 1108  P50: 9651.190476  P95: 14241.509434  P99: 20931.320755  Max: 46424.000000
    [1,0]<stderr>:OpsEmbeddingLookup statistics => count: 1106  P50: 261759.868421  P95: 310601.000000  P99: 310601.000000  Max: 310601.000000
    [1,0]<stderr>:OpsPushGrad statistics => count: 1107  P50: 41825.000000  P95: 49584.461967  P99: 65456.896552  Max: 71762.000000

`c++` 版本耗时统计如下。

    [1,0]<stdout>:OpsReadSample statistics => count: 757  P50: 3972.443182  P95: 302995.000000  P99: 302995.000000  Max: 302995.000000
    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 757  P50: 27567.438692  P95: 32672.547684  P99: 39838.000000  Max: 39838.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 758  P50: 7128.000000  P95: 9629.400000  P99: 9851.746667  Max: 10258.000000
    

`c++` 版本与 `rust` 版本对比如下。
备注: 计时单位为微妙。

| 统计项          | rust p50 | rust p95 | c++ p50 | c++ p95 | p50 speedup | p95 speedup |
|-----------------|----------|----------|---------|---------|-------------|-------------|
| ReadSample      | 9,651    | 14,241   | 3,972   | 302,995 | 2.43x       | 0.05x       |
| EmbeddingLookup | 261,759  | 310,601  | 27,567  | 32,672  | 9.50x       | 9.51x       |
| PushGrad        | 41,825   | 49,584   | 7,128   | 9,626   | 5.87x       | 5.15x       |

从主要的三项耗时统计来看，`ReadSample` 有 `2.43x` 的差距，可能是因为 `c++` 版本读取的是 `batch` 数据,
这种数据格式基本就是二进制的格式，直接按 `byte` 进行读取，速度非常快，而 `rust` 读取的是 `proto` 格式的
单条样本格式，还需要再拼 `batch`，速度会有差异。后面会尝试一种更新的可能比 `batch` 数据更好的格式。

更大的差异在于 `ps` 上的速度差异。从耗时可以看出，不管是 `c++` 版本还是 `rust` 版本，主要瓶颈都在
`EmbeddingLookup` 这一步。`c++` 版本快了 `9` 倍多。`ps` 上涉及的计算逻辑比较多，而且涉及到各种多线程
并发访问的问题，可能还是和具体实现有关系。

理论上 `rust` 的性能应该能和 `c++` 到统一水平, 实际 `rust` 差这么多有点出乎意料。

从实现上看差异比较大的主要是三个: 

- 保存参数的实现：
    - `rust`: 采用 `Vec<DashMap>` 来保存参数，固定分片，`shard` 会有锁，内部保存的数据是 `lock free`,
    并且 `map` 采用 `swiss table` 的思路实现，利用 `simd` 加速。
    - `c++`: 基于 `circular buffer` 自己实现了 `lru`，结合 `folly concurrent hashmap` 手动管理内存。
    这种实现会预分配一大片内存用户保存参数，不会有 `malloc` 和 `delete` 带来的问题。保存的数据也是
    `lock free`。
- `scheduler`:
    - `rust`: 直接使用了 `tokio` 提供的 `scheduler` 来调度并发任务。仔细看了 `tokio scheduler` 的原理，
    感觉和 `c++` 中自己实现的 `scheduler` 思路类似， 都是多个 `processors`, 每个都有自己的 `queue`,
    而且 `tokio` 的实现更完善一些，不应该有这么大差异。
    - `c++`: 自己实现的 `scheduler`。 思路是多个 `processors`, 每个都有自己的 `queue`。
- `grpc`:
    - `rust`: 直接使用 `tonic` 框架。
    - `c++`: 使用的是 `c++ grpc`, 同时封装了一些 `zero copy` 的优化。考虑到主要耗时的计算逻辑是
    `Embedding` 参数的查询与更新，`grpc` 请求 `qps` 并不是很高，因此 `grpc` 请求这一因素应该影响不大。

待仔细研究。

## 分析与总结

TODO:
- 仔细对比 `ps` 中的耗时监控。
- 仔细对比 `map` 的单线程读写性能。
- 仔细对比 `map` 的多线程读写性能。
