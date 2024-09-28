# `c++` 版本速度

`c++` 版本读取的是 `batch` 数据，格式为自定义的二进制格式，每行为一个 `batch` 的特征数据。

## 训练资源

- `trainer`: 1 T4 GPU, 20 core, 30G Mem, 1 副本。
- `ps`: 0 GPU, 60 core, 200G Mem, 4 副本。
- `hub`: 0 GPU, 20 core, 50G Mem, 4 副本。

## 训练速度

相同的资源情况下速度能到 `243848 examples/sec`, 特征一样，不过数据是拼好的 `batch`。

    [1,0]<stderr>:2024-09-21 10:28:35,027 - INFO [hooks.py:465 - after_run] - step 9600, auc = 0.8267, loss = 0.2872257071, prob_mean: 0.109512, real_mean: 0.125947, (238.1 it/sec; 243848.0 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:35,456 - INFO [hooks.py:465 - after_run] - step 9700, auc = 0.8267, loss = 0.2872743337, prob_mean: 0.109527, real_mean: 0.125963, (232.8 it/sec; 238371.1 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:35,889 - INFO [hooks.py:465 - after_run] - step 9800, auc = 0.8267, loss = 0.2872754446, prob_mean: 0.109525, real_mean: 0.125965, (231.1 it/sec; 236693.9 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:36,313 - INFO [hooks.py:465 - after_run] - step 9900, auc = 0.8267, loss = 0.2872792002, prob_mean: 0.109525, real_mean: 0.125968, (235.6 it/sec; 241291.7 examples/sec)
    [1,0]<stderr>:2024-09-21 10:28:36,739 - INFO [hooks.py:465 - after_run] - step 10000, auc = 0.8266, loss = 0.2873227067, prob_mean: 0.109530, real_mean: 0.125983, (234.9 it/sec; 240575.3 examples/sec)
    
    
查看耗时监控如下

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 757  P50: 27567.438692  P95: 32672.547684  P99: 39838.000000  Max: 39838.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 758  P50: 7128.000000  P95: 9629.400000  P99: 9851.746667  Max: 10258.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 757  P50: 3972.443182  P95: 302995.000000  P99: 302995.000000  Max: 302995.000000

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 2612  P50: 28552.437902  P95: 44828.169014  P99: 48997.558685  Max: 50970.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 2616  P50: 4062.656904  P95: 6443.281853  P99: 11396.500000  Max: 298651.000000
    [1,0]<stdout>:[2024-09-21 10:28:34.876] [info] [run_status.cc:72] [RunStatus] OpsEmbeddingLookup statistics => count: 6952  P50: 28703.185841  P95: 45343.420016  P99: 48266.000000  Max: 48266.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 6956  P50: 4365.349076  P95: 6430.344002  P99: 7487.526316  Max: 32105.000000
    
重新跑了一遍，速度有些差异，可能和集群跑的任务有关

速度

    [1,0]<stderr>:2024-09-24 20:51:40,494 - INFO [hooks.py:465 - after_run] - step 18900, auc = 0.8761, loss = 0.1924512684, prob_mean: 0.121220, real_mean: 0.101562, (193.4 it/sec; 198036.4 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:41,002 - INFO [hooks.py:465 - after_run] - step 19000, auc = 0.8761, loss = 0.2136482298, prob_mean: 0.139329, real_mean: 0.103516, (197.2 it/sec; 201916.2 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:41,503 - INFO [hooks.py:465 - after_run] - step 19100, auc = 0.8761, loss = 0.1577771753, prob_mean: 0.084985, real_mean: 0.074219, (199.4 it/sec; 204152.6 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:42,020 - INFO [hooks.py:465 - after_run] - step 19200, auc = 0.8762, loss = 0.1820228547, prob_mean: 0.102443, real_mean: 0.079102, (193.3 it/sec; 197945.0 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:42,534 - INFO [hooks.py:465 - after_run] - step 19300, auc = 0.8762, loss = 0.1943198144, prob_mean: 0.106581, real_mean: 0.090820, (194.6 it/sec; 199262.9 examples/sec)
    [1,0]<stderr>:2024-09-24 20:51:43,046 - INFO [hooks.py:465 - after_run] - step 19400, auc = 0.8762, loss = 0.1421866119, prob_mean: 0.096939, real_mean: 0.052734, (195.3 it/sec; 199998.4 examples/sec)


`trainer` 监控

    [1,0]<stdout>:OpsEmbeddingLookup statistics => count: 5508  P50: 27394.852941  P95: 44173.026316  P99: 56622.222222  Max: 363990.000000
    [1,0]<stdout>:OpsPushGrad statistics => count: 5512  P50: 9640.076336  P95: 13799.932461  P99: 19276.981132  Max: 23315.000000
    [1,0]<stdout>:OpsReadSample statistics => count: 5513  P50: 3959.264480  P95: 6492.352103  P99: 9530.824561  Max: 27105.000000

`ps` 耗时监控如下


    PsFeedSample statistics => count: 85098  P50: 85.895506  P95: 818.502277  P99: 1205.698832  Max: 2756.000000
    PsEmbeddingLookup statistics => count: 5684  P50: 6928.621908  P95: 9640.070671  P99: 9881.088339  Max: 19403.000000
    PsPushGrad statistics => count: 5683  P50: 8465.338310  P95: 12913.279570  P99: 13824.337243  Max: 24890.000000
    PsFeedCached statistics => count: 85111  P50: 332.000000  P95: 340.000000  P99: 340.000000  Max: 340.000000
    PsLookupCached statistics => count: 85061  P50: 27.978623  P95: 31.000000  P99: 31.000000  Max: 31.000000

`hub` 耗时监控如下

    HubReadSample statistics => count: 50  P50: 4.538462  P95: 28.000000  P99: 210000.000000  Max: 223220.000000
    HubNext statistics => count: 214  P50: 64375.000000  P95: 5302804.000000  P99: 5302804.000000  Max: 5302804.000000
    HubFeedSample statistics => count: 132  P50: 6780.000000  P95: 12658.181818  P99: 16035.000000  Max: 16035.000000
    HubCountSamplePos statistics => count: 27416  P50: 1.000000  P95: 1.000000  P99: 1.000000  Max: 1.000000
    HubCountSampleNeg statistics => count: 190724  P50: 1.000000  P95: 1.000000  P99: 1.000000  Max: 1.000000
    HubBatchProcessor statistics => count: 194  P50: 3030.434783  P95: 7634394.000000  P99: 7634394.000000  Max: 7634394.000000

